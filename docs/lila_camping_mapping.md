# LILA 露营类别映射概览

本表是固定映射规则，不表示每个类别都有有效训练样本。实际入选数量、整图排除及划分由运行后的 AUDIT_REPORT.md 统计。仅处理五个 LILA 数据集。

## 英文类别与展示分组

| 英文类别 | 中文含义 | 展示分组 | 展示父类 |
| --- | --- | --- | --- |
| aardvark | 土豚 | other_mammal | — |
| antelope | 羚羊类 | ungulate | — |
| baboon | 狒狒 | primate | monkey |
| badger | 獾类（含鼬獾、蜜獾） | small_carnivore | — |
| bamboo_rat | 竹鼠 | rodent | rodent |
| bat | 蝙蝠 | other_mammal | — |
| bear | 熊（不细分） | bear | — |
| bird | 鸟类（不细分） | bird | — |
| black_bear | 黑熊 | bear | bear |
| buffalo | 非洲水牛 | ungulate | — |
| cat | 家猫 | cat_family | — |
| cattle | 家牛 | ungulate | — |
| chevrotain | 鼷鹿类 | ungulate | — |
| chicken | 家鸡 | bird | bird |
| chipmunk | 花鼠 | rodent | rodent |
| civet | 灵猫类（通俗分组） | small_carnivore | — |
| cougar | 美洲狮 | cat_family | — |
| coyote | 郊狼 | canid | — |
| crow | 乌鸦 | bird | bird |
| deer | 鹿类 | ungulate | — |
| dog | 家犬 | canid | — |
| eagle | 鹰类 | bird | bird |
| elephant | 象 | ungulate | — |
| elk | 北美马鹿 | ungulate | deer |
| flying_squirrel | 鼯鼠 | rodent | rodent |
| fox | 狐（不细分） | canid | — |
| galago | 婴猴类 | primate | — |
| gibbon | 长臂猿 | primate | — |
| goat | 家山羊 | ungulate | — |
| horse | 马 | ungulate | — |
| hyena | 鬣狗 | carnivore | — |
| insect | 昆虫 | invertebrate | — |
| jackal | 胡狼 | canid | — |
| leopard | 豹 | cat_family | — |
| leopard_cat | 豹猫 | cat_family | — |
| lizard | 蜥蜴 | reptile | — |
| lynx | 猞猁类（含短尾猫） | cat_family | — |
| magpie | 鹊类 | bird | bird |
| marmot | 旱獭 | rodent | rodent |
| mink | 水貂 | small_carnivore | mustelid |
| mongoose | 獴类 | small_carnivore | — |
| monitor_lizard | 巨蜥 | reptile | lizard |
| monkey | 猴类 | primate | — |
| mouse | 鼠类（含林鼠） | rodent | rodent |
| mustelid | 其他鼬科动物（含貂） | small_carnivore | — |
| opossum | 负鼠 | other_mammal | — |
| otter | 水獭 | small_carnivore | mustelid |
| owl | 鸮类 | bird | bird |
| pangolin | 穿山甲 | other_mammal | — |
| person | 人 | non_animal | — |
| pheasant | 雉鸡类（通俗分组） | bird | bird |
| pig | 猪（未明确家野） | pig | — |
| pigeon | 鸠鸽类 | bird | bird |
| porcupine | 豪猪 | rodent | rodent |
| rabbit | 兔类 | other_mammal | — |
| raccoon | 浣熊 | other_mammal | — |
| red_fox | 赤狐 | canid | fox |
| rodent | 其他啮齿动物 | rodent | — |
| serow | 鬣羚 | ungulate | — |
| skunk | 臭鼬 | small_carnivore | — |
| snake | 蛇 | reptile | — |
| squirrel | 松鼠类 | rodent | rodent |
| tree_shrew | 树鼩 | other_mammal | — |
| turkey | 火鸡 | bird | bird |
| turtle | 龟鳖类 | reptile | — |
| vehicle | 车辆 | non_animal | — |
| warthog | 疣猪 | pig | pig |
| weasel | 鼬类 | small_carnivore | mustelid |
| wild_boar | 野猪（含非洲丛林猪） | pig | pig |
| wild_cat | 其他野生小型猫科 | cat_family | — |
| woodpecker | 啄木鸟 | bird | bird |
| zebra | 斑马 | ungulate | — |

## swg-camera-traps

原 ID 为当前原始文件的参考 ID；程序按数据集和原始名称匹配。排除类别触发整图排除。

| 参考原 ID | 原始名称 | 原始中文 | 输出英文类别 |
| --- | --- | --- | --- |
| 0 | empty | 空图（未标记动物） | 排除 |
| 1 | ignore | 忽略标签 | 排除 |
| 2 | blurred | 模糊图像 | 排除 |
| 4 | crab_eating_mongoose | 食蟹獴 | mongoose |
| 5 | ferret_badger | 鼬獾类（未定种） | badger |
| 6 | large_antlered_muntjac | 大角麂 | deer |
| 7 | stump_tailed_macaque | 熊猴（短尾猴） | monkey |
| 8 | unidentified_macaque | 未识别猕猴 | monkey |
| 9 | unidentified_muntjac | 未识别麂类 | deer |
| 10 | unidentified_small_mammal | 未识别小型哺乳动物 | 排除 |
| 11 | unidentified_squirrel | 未识别松鼠 | squirrel |
| 12 | eurasian_wild_pig | 欧亚野猪 | wild_boar |
| 13 | problem | 问题图像／异常标签 | 排除 |
| 14 | common_palm_civet | 椰子狸 | civet |
| 15 | silver_pheasant | 白鹇 | pheasant |
| 16 | unidentified_palm_civet | 未识别棕榈狸类 | civet |
| 17 | insect | 昆虫类 | insect |
| 18 | masked_palm_civet | 果子狸（花面狸） | civet |
| 19 | pig_tailed_macaque | 豚尾猴 | monkey |
| 20 | red_cheeked_squirrel | 红颊长吻松鼠 | squirrel |
| 21 | roosevelts_muntjac_group | 罗斯福麂类群 | deer |
| 22 | sambar | 水鹿 | deer |
| 23 | spotted_linsang | 斑林狸 | civet |
| 24 | unidentified_animal | 未识别动物 | 排除 |
| 25 | unidentified_bird | 未识别鸟类 | bird |
| 26 | unidentified_partridge | 未识别鹧鸪类 | bird |
| 27 | unidentified_ungulates | 未识别有蹄类 | 排除 |
| 28 | unidentified_murid | 未识别鼠科动物 | mouse |
| 29 | red_junglefowl | 红原鸡 | bird |
| 30 | blue_pitta | 蓝八色鸫 | bird |
| 31 | pallass_squirrel | 赤腹松鼠 | squirrel |
| 32 | yellow_throated_marten | 黄喉貂 | bird |
| 33 | grey_peacock_pheasant | 灰孔雀雉 | pheasant |
| 34 | white_crowned_forktail | 白冠燕尾 | bird |
| 35 | emerald_dove | 绿翅金鸠 | pigeon |
| 36 | human | 人 | person |
| 37 | unidentified_pitta | 未识别八色鸫 | bird |
| 38 | laughingthrush | 噪鹛类（未定种） | bird |
| 39 | asiatic_brush_tailed_porcupine | 亚洲帚尾豪猪 | porcupine |
| 40 | large_indian_civet | 大灵猫 | civet |
| 41 | unidentified_otter | 未识别水獭 | otter |
| 42 | unidentified_owl | 未识别鸮类（猫头鹰） | owl |
| 43 | white-winged_magpie | 白翅蓝鹊 | magpie |
| 44 | japanese_thrush | 乌灰鸫 | bird |
| 45 | orange_headed_thrush | 橙头地鸫 | bird |
| 46 | scaly_thrush | 虎斑地鸫 | bird |
| 47 | unidentified_mammal | 未识别哺乳动物 | 排除 |
| 48 | gray_laughingthrush | 褐胸噪鹛 | bird |
| 49 | eurasian_woodcock | 丘鹬 | bird |
| 50 | red_collared_woodpecker | 红领绿啄木鸟 | woodpecker |
| 51 | blue_whistling_thrush | 紫啸鸫 | bird |
| 52 | large_scimitar_babbler | 长嘴钩嘴鹛 | bird |
| 53 | rufous_cheeked_laughingthrush | 栗颊噪鹛 | bird |
| 54 | chinese_serow | 中华鬣羚（按原英文名） | serow |
| 55 | gibbon | 长臂猿类（未定种） | gibbon |
| 56 | red_shanked_douc | 白臀叶猴（红腿白臀叶猴） | monkey |
| 57 | annamite_striped_rabbit | 安南条纹兔 | rabbit |
| 58 | blue_fronted_robin | 蓝额长脚地鸲 | bird |
| 59 | orange_breasted_trogon | 橙胸咬鹃 | bird |
| 60 | chevrotain | 鼷鹿类（未定种） | chevrotain |
| 61 | unidentified_pheasant | 未识别雉类 | pheasant |
| 62 | bar_bellied_pitta | 斑腹八色鸫 | bird |
| 63 | domestic_dog | 家犬 | dog |
| 64 | malayan_porcupine | 马来豪猪 | porcupine |
| 65 | assam_or_rhesus_macaque | 阿萨姆猴或恒河猴 | monkey |
| 66 | unidentified_bat | 未识别蝙蝠 | bat |
| 67 | indochinese_green_magpie | 印支绿鹊 | magpie |
| 68 | common_green_magpie | 蓝绿鹊 | magpie |
| 69 | francois_langur | 黑叶猴 | monkey |
| 70 | leopard_cat | 豹猫 | leopard_cat |
| 71 | binturong | 熊狸 | civet |
| 72 | hog_badger | 猪獾 | badger |
| 73 | crested_argus | 冠眼斑雉 | pheasant |
| 74 | pangolin | 穿山甲类（未定种） | pangolin |
| 75 | bamboo_rat | 竹鼠类（未定种） | bamboo_rat |
| 76 | asiatic_black_bear | 亚洲黑熊 | black_bear |
| 77 | mountain_hawk_eagle | 鹰雕 | eagle |
| 78 | unidentified_bear | 未识别熊类 | bear |
| 79 | unidentified_weasel | 未识别鼬类 | weasel |
| 80 | northern_treeshrew | 北树鼩 | tree_shrew |
| 81 | bengal_monitor | 孟加拉巨蜥 | monitor_lizard |
| 82 | black_giant_squirrel | 巨松鼠（黑巨松鼠） | squirrel |
| 83 | sun_bear | 马来熊 | bear |
| 84 | marbled_cat | 云猫 | wild_cat |
| 85 | unidentified_striped_squirrel | 未识别条纹松鼠 | squirrel |
| 86 | hatinh_langur | 河静叶猴 | monkey |
| 87 | unidentified_turtles | 未识别龟鳖类 | turtle |
| 88 | red_muntjac | 赤麂（红麂） | deer |
| 89 | phayres_langur | 菲氏叶猴 | monkey |
| 90 | giant_flying_squirrel | 大鼯鼠类（未定种） | flying_squirrel |
| 91 | rufous_necked_hornbill | 棕颈犀鸟 | bird |
| 92 | golden_cat | 亚洲金猫 | wild_cat |
| 93 | owstons_civet | 缟灵猫 | civet |
| 94 | invertebrate | 无脊椎动物 | 排除 |
| 95 | chinese serow | 中华鬣羚（按原英文名） | serow |
| 96 | unidentified_lizard | 未识别蜥蜴 | lizard |
| 97 | macaque_not_stump_tailed | 非熊猴的猕猴类 | monkey |
| 98 | unidentified_small_animal | 未识别小型动物 | 排除 |
| 99 | yellow_bellied_weasel | 黄腹鼬 | weasel |
| 100 | brown_wood_owl | 褐林鸮 | owl |
| 101 | necklaced_laughingthrush_sp | 领纹噪鹛类（未定种） | bird |
| 102 | mountain_imperial_pigeon | 山皇鸠 | pigeon |
| 103 | rufous_eared_laughingthrush | 棕耳噪鹛（原英文暂译） | bird |
| 104 | lesser_yellownape | 黄冠啄木鸟 | woodpecker |
| 105 | red_tailed_laughingthrush | 赤尾噪鹛 | bird |
| 106 | flying_squirrel | 鼯鼠类（飞鼠） | flying_squirrel |
| 107 | siberian_thrush | 白眉地鸫 | bird |
| 108 | asian_fairy_bluebird | 和平鸟 | bird |
| 109 | forktail_sp | 燕尾类鸟（未定种） | bird |
| 110 | striated_heron | 绿鹭 | bird |
| 111 | crested_serpent_eagle | 蛇雕 | eagle |
| 112 | black_throated_laughingthrush | 黑喉噪鹛 | bird |
| 113 | golden_babbler | 金头穗鹛 | bird |
| 114 | back_striped_weasel | 背纹鼬 | weasel |
| 115 | barred_cuckoo_dove | 斑尾鹃鸠 | pigeon |
| 116 | unidentified_snake | 未识别蛇类 | snake |
| 117 | white_tailed_robin | 白尾蓝地鸲 | bird |
| 118 | eyebrowed_thrush | 白眉鸫 | bird |
| 119 | silver_eared_mesia | 银耳相思鸟 | bird |
| 120 | grey_headed_canary_flycatcher | 方尾鹟 | bird |
| 121 | vehicle | 车辆 | vehicle |

## caltech-camera-traps

原 ID 为当前原始文件的参考 ID；程序按数据集和原始名称匹配。排除类别触发整图排除。

| 参考原 ID | 原始名称 | 原始中文 | 输出英文类别 |
| --- | --- | --- | --- |
| 6 | bobcat | 短尾猫（山猫） | lynx |
| 1 | opossum | 负鼠类 | opossum |
| 30 | empty | 空图（未标记动物） | 排除 |
| 9 | coyote | 郊狼 | coyote |
| 3 | raccoon | 浣熊 | raccoon |
| 11 | bird | 鸟类（未定种） | bird |
| 8 | dog | 家犬 | dog |
| 16 | cat | 家猫 | cat |
| 5 | squirrel | 松鼠类（未定种） | squirrel |
| 10 | rabbit | 兔类（未定种） | rabbit |
| 7 | skunk | 臭鼬类（未定种） | skunk |
| 14 | lizard | 蜥蜴类（未定种） | lizard |
| 99 | rodent | 啮齿动物（未定种） | rodent |
| 21 | badger | 獾类 | badger |
| 34 | deer | 鹿类（未定种） | deer |
| 37 | cow | 家牛 | cattle |
| 33 | car | 汽车 | vehicle |
| 51 | fox | 狐类 | fox |
| 39 | pig | 猪类 | pig |
| 40 | mountain_lion | 美洲狮 | cougar |
| 66 | bat | 蝙蝠类（未定种） | bat |
| 97 | insect | 昆虫类 | insect |

## ena24

原 ID 为当前原始文件的参考 ID；程序按数据集和原始名称匹配。排除类别触发整图排除。

| 参考原 ID | 原始名称 | 原始中文 | 输出英文类别 |
| --- | --- | --- | --- |
| 0 | Bird | 鸟类（未定种） | bird |
| 1 | Eastern Gray Squirrel | 东部灰松鼠 | squirrel |
| 2 | Eastern Chipmunk | 东部花栗鼠 | chipmunk |
| 3 | Woodchuck | 美洲旱獭 | marmot |
| 4 | Wild Turkey | 野火鸡 | turkey |
| 5 | White_Tailed_Deer | 白尾鹿 | deer |
| 6 | Virginia Opossum | 弗吉尼亚负鼠 | opossum |
| 7 | Eastern Cottontail | 东部棉尾兔（佛罗里达棉尾兔） | rabbit |
| 8 | Human | 人 | person |
| 9 | Vehicle | 车辆 | vehicle |
| 10 | Striped Skunk | 条纹臭鼬 | skunk |
| 11 | Red Fox | 赤狐 | red_fox |
| 12 | Eastern Fox Squirrel | 东部狐松鼠 | squirrel |
| 13 | Northern Raccoon | 北美浣熊 | raccoon |
| 14 | Grey Fox | 灰狐 | fox |
| 15 | Horse | 家马 | horse |
| 16 | Dog | 家犬 | dog |
| 17 | American Crow | 短嘴鸦（美洲乌鸦） | crow |
| 18 | Chicken | 家鸡 | chicken |
| 19 | Domestic Cat | 家猫 | cat |
| 20 | Coyote | 郊狼 | coyote |
| 21 | Bobcat | 短尾猫（山猫） | lynx |
| 22 | American Black Bear | 美洲黑熊 | black_bear |

## oregon-critters

原 ID 为当前原始文件的参考 ID；程序按数据集和原始名称匹配。排除类别触发整图排除。

| 参考原 ID | 原始名称 | 原始中文 | 输出英文类别 |
| --- | --- | --- | --- |
| 0 | empty | 空图（未标记动物） | 排除 |
| 1 | coyote | 郊狼 | coyote |
| 2 | bat species | 蝙蝠类（未定种） | bat |
| 3 | strigidae family | 鸱鸮科鸟类 | owl |
| 4 | ruffed grouse | 披肩榛鸡 | bird |
| 5 | red fox | 赤狐 | red_fox |
| 6 | canada jay | 加拿大噪鸦 | bird |
| 7 | pileated woodpecker | 北美黑啄木鸟 | woodpecker |
| 8 | reptile or amphibian | 爬行动物或两栖动物 | 排除 |
| 9 | bobcat | 短尾猫（山猫） | lynx |
| 10 | marten | 貂类 | mustelid |
| 11 | california ground squirrel | 加利福尼亚地松鼠 | squirrel |
| 12 | mink | 水貂类 | mink |
| 13 | western spotted skunk | 西部斑臭鼬 | skunk |
| 14 | mountain beaver | 山河狸 | rodent |
| 15 | black bear | 黑熊 | black_bear |
| 16 | black-tailed deer | 黑尾鹿 | deer |
| 17 | cougar | 美洲狮 | cougar |
| 18 | roosevelt elk | 罗斯福马鹿 | elk |
| 19 | stellers jay | 冠蓝鸦（斯特勒松鸦） | bird |
| 20 | catharus species | 夜鸫属鸟类（未定种） | bird |
| 21 | varied thrush | 杂色鸫 | bird |
| 22 | dark-eyed junco | 暗眼灯草鹀 | bird |
| 23 | american robin | 旅鸫（美洲知更鸟） | bird |
| 24 | leporidae family | 兔科动物 | rabbit |
| 25 | douglas squirrel | 道格拉斯松鼠 | squirrel |
| 26 | humboldt's flying squirrel | 洪堡鼯鼠 | flying_squirrel |
| 27 | townsend's chipmunk | 汤氏花鼠 | chipmunk |
| 28 | weasel family | 鼬科动物 | mustelid |
| 29 | parulidae family | 森莺科鸟类 | bird |
| 30 | invertebrate | 无脊椎动物 | 排除 |
| 31 | picidae family | 啄木鸟科鸟类 | woodpecker |
| 32 | neotoma species | 林鼠属动物（未定种） | mouse |
| 33 | poecile species | 山雀类（Poecile 属，未定种） | bird |
| 34 | virginia opossum | 弗吉尼亚负鼠 | opossum |
| 35 | troglodytidae family | 鹪鹩科鸟类 | bird |
| 36 | mountain quail | 山鹑（北美山鹑） | bird |
| 37 | northern raccoon | 北美浣熊 | raccoon |
| 38 | striped skunk | 条纹臭鼬 | skunk |
| 39 | western gray squirrel | 西部灰松鼠 | squirrel |
| 40 | northern flicker | 金翼啄木鸟 | woodpecker |
| 41 | domestic dog | 家犬 | dog |
| 42 | human | 人 | person |
| 43 | small mammal | 小型哺乳动物（未定种） | 排除 |
| 44 | gray fox | 灰狐 | fox |
| 45 | other bird | 其他鸟类 | bird |

## nkhotakota-camera-traps

原 ID 为当前原始文件的参考 ID；程序按数据集和原始名称匹配。排除类别触发整图排除。

| 参考原 ID | 原始名称 | 原始中文 | 输出英文类别 |
| --- | --- | --- | --- |
| 0 | empty | 空图（未标记动物） | 排除 |
| 1 | human | 人 | person |
| 2 | other_animal | 其他动物（未定类） | 排除 |
| 3 | african_savanna_elephant | 非洲草原象 | elephant |
| 4 | sable | 黑马羚 | antelope |
| 5 | yellow_baboon | 黄狒狒 | baboon |
| 6 | bushpig | 非洲丛林猪 | wild_boar |
| 7 | greater_kudu | 大捻角羚 | antelope |
| 8 | southern_reedbuck | 南苇羚 | antelope |
| 9 | bird | 鸟类（未定种） | bird |
| 10 | small_antelope | 小型羚羊（未定种） | antelope |
| 11 | cape_bushbuck | 开普薮羚 | antelope |
| 12 | cape_porcupine | 南非豪猪（开普豪猪） | porcupine |
| 13 | common_duiker | 灰小羚羊 | antelope |
| 14 | common_warthog | 普通疣猪 | warthog |
| 15 | aardvark | 土豚 | aardvark |
| 16 | common_waterbuck | 普通水羚 | antelope |
| 17 | southern_ground_hornbill | 南方地犀鸟 | bird |
| 18 | cape_buffalo | 开普水牛 | buffalo |
| 19 | african_civet | 非洲灵猫 | civet |
| 20 | leopard | 豹 | leopard |
| 21 | galago | 婴猴类（未定种） | galago |
| 22 | honey_badger | 蜜獾 | badger |
| 23 | mongoose | 獴类（未定种） | mongoose |
| 24 | genet | 獛类（未定种） | civet |
| 25 | red_duiker | 红小羚羊 | antelope |
| 26 | common_eland | 大羚羊（伊兰羚羊） | antelope |
| 27 | plains_zebra | 平原斑马 | zebra |
| 28 | impala | 黑斑羚 | antelope |
| 29 | roan | 马羚 | antelope |
| 30 | klipspringer | 岩羚 | antelope |
| 31 | spotted_hyena | 斑鬣狗 | hyena |
| 32 | serval | 薮猫 | wild_cat |
| 33 | vervet_monkey | 黑长尾猴（绿猴） | monkey |
| 34 | domestic_dog | 家犬 | dog |
| 35 | sharpes_grysbok | 夏氏小羚羊 | antelope |
| 36 | helmeted_guineafowl | 盔珠鸡 | bird |
| 37 | scrub_hare | 灌丛野兔 | rabbit |
| 38 | small_mammal | 小型哺乳动物（未定种） | 排除 |
| 39 | side-striped_jackal | 侧纹胡狼 | jackal |
| 40 | palm_civet | 棕榈狸类 | civet |
| 41 | squirrel | 松鼠类（未定种） | squirrel |
| 42 | bat | 蝙蝠类（未定种） | bat |
| 43 | blue_monkey | 青长尾猴 | monkey |
| 44 | domestic_cattle | 家牛 | cattle |
| 45 | domestic_goat | 家山羊 | goat |
