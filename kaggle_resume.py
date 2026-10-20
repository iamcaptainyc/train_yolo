"""Resume Kaggle dataset bundles while preserving incomplete .archive files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile

STATE_FILE = '.kaggle-resume.json'


def transfer_archive(response, archive: Path):
    """Keep partial bytes on errors; append only a correctly aligned HTTP 206."""
    import requests
    from tqdm import tqdm

    try:
        response.raise_for_status()
        if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
            raise ValueError('Encoded download cannot safely resume; expected identity transfer encoding')
        total = int(response.headers.get('Content-Length', 0))
        expected_hash = response.headers.get('X-Goog-Hash', '')
        offset = archive.stat().st_size if archive.exists() else 0
        if total and offset > total:
            raise ValueError(f'Partial archive is larger than remote archive: {archive}; inspect it before restarting')
        if not total or offset != total:
            if offset:
                url = response.url
                response.close()
                response = requests.get(url, headers={'Range': f'bytes={offset}-', 'Accept-Encoding': 'identity'},
                                        stream=True, timeout=(30, 120))
                response.raise_for_status()
                if response.status_code == 206:
                    content_range = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not content_range or int(content_range[1]) != offset:
                        raise ValueError('Invalid Content-Range; refused to append mismatched bytes')
                    range_total = int(content_range[3])
                    if total and total != range_total:
                        raise ValueError('Remote archive size changed; refused to mix versions')
                    if int(content_range[2]) != range_total - 1:
                        raise ValueError('Server returned an incomplete byte range')
                    total = range_total
                    print(f'Resuming {archive.name} from {offset:,} bytes', flush=True)
                elif response.status_code == 200:
                    raise ValueError('Server ignored the Range request; partial archive preserved. '
                                     'Use a new output directory to restart without deleting existing progress.')
                else:
                    raise ValueError(f'Unexpected HTTP status for resume: {response.status_code}')
                if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                    raise ValueError('Encoded byte-range response cannot safely resume')
            with archive.open('ab' if offset else 'wb') as stream, tqdm(
                total=total or None, initial=offset, unit='B', unit_scale=True, desc=archive.name,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        progress.update(len(chunk))
            if total and archive.stat().st_size != total:
                raise requests.exceptions.ConnectionError('Incomplete archive; partial bytes preserved for retry')
        # GCS advertises a base64 MD5 when available. Never assume equal sizes imply valid bytes.
        expected_md5 = next((v.strip()[4:] for v in expected_hash.split(',') if v.strip().startswith('md5=')), None)
        if expected_md5:
            import base64
            digest = hashlib.md5()
            with archive.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if base64.b64encode(digest.digest()).decode() != expected_md5:
                raise ValueError(f'Archive MD5 mismatch: {archive}; preserved for inspection, use a new directory to restart')
    finally:
        response.close()


def extract_archive(archive: Path, output: Path):
    """Validate ZIP CRC and every member path before modifying extracted files."""
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (output / member.filename).resolve()
            if output not in target.parents or target in (archive.resolve(), output / STATE_FILE,
                                                          output / (STATE_FILE + '.tmp')):
                raise ValueError(f'Invalid or reserved archive member: {member.filename}')
            if target == output / '.complete' or output / '.complete' in target.parents:
                raise ValueError(f'Reserved completion metadata path: {member.filename}')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Symlink archive member: {member.filename}')
        bad = bundle.testzip()
        if bad:
            raise ValueError(f'Archive CRC check failed: {bad}; partial file preserved at {archive}')
        bundle.extractall(output)


def resumable_dataset_download(handle, *, output_dir, force_download=False, path=None):
    """Same calling convention as kagglehub.dataset_download, for full bundles only.

    Keep KaggleHub/SDK-specific calls here; our own transfer avoids its nonempty-dir
    guard and does not delete partial downloads on retries or process restarts.
    """
    if force_download or path is not None:
        raise ValueError('Resume mode requires a whole dataset without --force or --file')
    from kagglehub.cache import Cache
    from kagglehub.clients import build_kaggle_client
    from kagglehub.exceptions import handle_call
    from kagglehub.handle import parse_dataset_handle
    from kagglehub.http_resolver import _build_dataset_download_request, _get_current_version

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / STATE_FILE
    h = parse_dataset_handle(handle)
    identity = f'{h.owner}/{h.dataset}'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if state:
        if state.get('dataset') != identity or (h.is_versioned() and h.version != state.get('version')):
            raise ValueError('Resume metadata belongs to another dataset or version; use a separate directory')
        h = h.with_version(int(state['version']))
    elif not h.is_versioned():
        # Migrate an incomplete direct-output download from earlier script versions.
        versions = {int(p.stem) for p in output.glob('*.archive') if p.stem.isdigit()}
        markers = output / '.complete' / 'datasets' / h.owner / h.dataset
        if markers.is_dir():
            versions.update(int(p.name) for p in markers.iterdir() if p.name.isdigit() and (p/'bundle.complete').exists())
        if len(versions) > 1:
            raise ValueError('Several versions found; specify OWNER/DATASET/versions/N to resume one')
        if versions:
            h = h.with_version(versions.pop())
    with build_kaggle_client() as api_client:
        if not h.is_versioned():
            h = h.with_version(_get_current_version(api_client, h))
        cache = Cache(override_dir=str(output))
        if cache.load_from_cache(h):
            print(f'[{identity}] Already complete: {output}', flush=True)
            return str(output)
        archive = Path(cache.get_archive_path(h))
        if any(output.iterdir()) and not (state or archive.is_file()):
            raise ValueError(f'No resumable {archive.name} or matching metadata in {output}; '
                             'existing extracted files alone cannot resume an archive download')
        state = {'dataset': identity, 'version': h.version}
        temporary = output / (STATE_FILE + '.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(state_path)
        request = _build_dataset_download_request(h, None)
        response = handle_call(lambda: api_client.datasets.dataset_api_client.download_dataset(request), h)
        transfer_archive(response, archive)
        print(f'[{identity}] Validating and extracting {archive.name}...', flush=True)
        extract_archive(archive, output)
        cache.mark_as_complete(h)
        archive.unlink()
        return str(output)
