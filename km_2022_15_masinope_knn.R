# Kvantitatiivsed meetodid
# Masinõpe KNN näitel paketiga mlr
# 7. detsember 2022

library(tidyverse)
library(mlr)

# loeme andmed sisse

spambase <- read.csv("data/spambase.csv")

# vaatame tunnuste jaotusparameetreid

psych::describe(spambase)

# tunnused on asümmeetrilise jaotusega ja suurte erinditega. Kas tuleks tunnuste jaotuseid teisendamisega normaaljaotusele lähemale tuua? Erindid võivad klassifitseerimise tulemusi mõjutada, aga kui erindid esinevad reeglina spämmikirjade seas ja tavameilid on reeglina 0-väärtuste ja selle lähedaste väärtuste seas, siis ei pruugi see vajalik olla. kNN meetod iseenesest tunnuste jaotusele eeldusi ei sea. Proovime teha klassifitseerimise läbi ilma tunnuseid normaaljaotusele vastavaks teisendamata, küll aga peaks tunnuste skaalasid muutma, sest need on mõnel juhul väga erinevad. Enne veel kontrollime klassifitseeriva tunnuse tüüpi ja muudame selle kategoriaalseks ehk factoriks, sest see tunnus peab klassifitseerimiseks olema kategoriaalne.

class(spambase$cat)
spambase$cat <- as.factor(spambase$cat)

spambase <- spambase %>% 
  mutate(across(where(is.numeric), scale)) %>% 
  mutate(across(where(is.matrix), as.numeric)) 


# eraldame andmed mudeli õpetamiseks (treenimiseks) ja testimiseks. Sisselaetud andmestikus on juba eelnevalt ridade järjekord juhuslikustatud, nii et saame siin andmed kaheks eraldada lihtsalt reanumbrite järgi.

spambase_trn <- spambase %>% 
  slice(1:3680)

spambase_tst <- spambase %>% 
  slice(3681:4601)


# kasutame masinõppe meetodeid paketi mlr abil, millega saab R-s hõlpsalt KNN-i ja muid masinõppemeetodeid (sh tavalisi regressioonimudeleid) kasutada. Paketi loogika erinevate masinõppemeetodite puhul on, et kõigepealt paneme paika ülesande, mida tahame masinõppega lahendada - antud juhul teatud emaile kirjeldavate tunnuste põhjal klassifitseerida spämmiks ja mittespämmiks

spamTask <- makeClassifTask(data = spambase_trn, target = "cat")

# defineerime õpimeetodi ehk valime algoritmi, mida klassifitseerimisel kasutame koos vajalike argumentidega

knn <- makeLearner("classif.knn", k = 11)

# infoks: milliseid õpimeetodeid mlr veel võimaldab?

listLearners()$class
listLearners("classif")$class

# treenime mudeli - rakendame eelnevalt defineeritud meetodi sõnastatud ülesandele ja saame mudeli, millega on pmst võimalik tulevikus uute andmete põhjal järeldusi teha (uute meilide puhul otsustada, kas tegu on spämmiga või mitte). Treenimine käib paketis mlr funktsiooniga train, mille esimene argument on õpimeetod (vastav objekt eelnevalt loodud funktsooniga makeLearner), teine argument ülesanne (loodud funktsiooniga makeClassifTask, millega eelnevalt defineerisime andmed ja klassi tunnuse).

knnModel <- train(knn, spamTask)

# loodud mudelit saame kasutada uute / muude andmete puhul otsustamaks, milline email on spämm ja milline mitte. Spämmistaatuse prognoosimine käib sel juhul funktsiooniga `predict`. Antud juhul on meil testandmetes ka spämmistaatuse tegelik väärtus, st andmestiku loojate poolt kodeerimise põhjal saadud spämmistaatus, seetõttu näidatakse meile tulemusena mitte ainult prognoose, vaid ka tunnuse `cat` väärtusi testandmestikus.

knnProgn <- predict(knnModel, newdata = spambase_tst)
knnProgn

# uurime mudeli täpsust testandmete põhjal

table(knnProgn$data$truth, knnProgn$data$response)
descr::crosstab(knnProgn$data$truth, knnProgn$data$response, prop.c = T)

performance(knnProgn, measures = list(mmce, acc))

#### Valideerimine ####

# holdout validation - pmst seda juba tegime, mlr funktsioonid annavad ehk mugavamaid võimalusi (nt klassifitseeriva tunnuse alusel kihistamine)

holdout <- makeResampleDesc(method = "Holdout", split = 4/5, stratify = T)

# defineerime uue ülesande, sest eelnevalt jaotasime ise andmestiku kaheks osaks ja õppeülesandeks valisime ainult ühe osa algsest andmestikust, siin saame ülesandes ära määratleda kogu andmestiku ja selle jaotamine õppe- ja testvalimiks toimub valideerimise käigus

spamTask_valid <- makeClassifTask(data = spambase, target = "cat")

holdoutvalid <- resample(learner = knn, task = spamTask_valid, resampling = holdout, measures = list(mmce, acc))

View(holdoutvalid)
holdoutvalid$aggr
holdoutvalid$aggr[2]

calculateConfusionMatrix(holdoutvalid$pred, relative = T)

# k-fold CV, k = 5

kFold5valid <- resample(learner = knn, task = spamTask_valid, resampling = cv5, measures = list(mmce, acc))

# eelnevas käsus on argumendi resampling väärtuse cv5 näol tegu sisseehitatud k-fold valideerimise valikuskeemiga, kus andmestik jaotatakse k = 5 osaks, analoogsed on cv2, cv3, cv10. Kui tahame anda k-le muu väärtuse, tuleb eelnevalt defineerida valikuskeem funktsiooniga makeResampleDesc (nagu eelnevalt ühekordse valideerimise juures). Valideerimise saab lisaks läbi teha mitu korda, siis on tegu n-ö repeated k-fold CV-ga (iter = folds * reps), vaikeseadena reps = 10.

kFold5_4 <- makeResampleDesc(method = "RepCV", folds = 5, reps = 4, stratify = T)

kFold5_4valid <- resample(learner = knn, task = spamTask_valid, resampling = kFold5_4, measures = list(mmce, acc))

calculateConfusionMatrix(kFold5_4valid$pred, relative = T)


# LOOCV

LOO <- makeResampleDesc(method = "LOO")

# järgnev käsk võtab umbes paar minutit, nii et käivitage omal riisikol :)
LOOvalid <- resample(learner = knn, task = spamTask_valid, resampling = LOO, measures = list(mmce, acc))

calculateConfusionMatrix(LOOCV$pred, relative = T)

#### Optimeerimine ####

# Eelnevast mudeli koostamisest saab õppe kontekstis rääkida vaid tinglikult - andsime ise ette k väärtuse ehk mitme lähima naabri liigikuuluvuse alusel indiviid klassifitseeritakse. Püüame nüüd ka mudelit optimeerida ehk leida hüperparameetri väärtuse, mis annaks meile mudeli, mis võimaldaks klassikuuluvust prognoosida võimalikult täpselt, kuid samas ei oleks mudel üle sobitatud.

# Defineerime, milliseid k (k nagu kNN, mitte k-fold) väärtusi katsetame
knnParamSpace <- makeParamSet(makeDiscreteParam("k", values = 3:30))

# Paneme paika, kuidas just defineeritud k-de hulgast (väärtused 3 kuni 20) erinevaid väärtusi optimeerimisel otsitakse. Antud valik on väga lihtne - proovitakse läbi kõik k väärtused. Kui hüperparameetreid on mitu ja neil kõigil palju erinevaid võimalikke väärtuseid, ei pruugi see mõttekas olla, aga antud juhul on see valik ok.
gridSearch <- makeTuneControlGrid()

# Optimeerime mudelit ehk n-ö tuunime hüperparameetrit - teeme klassifitseerimise läbi, katsetades k (nagu kNN) väärtusi kolmest 20-ni ning valideerime iga k puhul tulemust k-fold valideerimisega, kus andmestik on jaotatud viieks osaks.

OptimKNN <- tuneParams("classif.knn", task = spamTask_valid, resampling = cv5, par.set = knnParamSpace, control = gridSearch)

# teeme optimeerimise joonise ka

OptimKNNres <- generateHyperParsEffectData(OptimKNN)

plotHyperParsEffect(OptimKNNres, x = "k", y = "mmce.test.mean", plot.type = "line")

# saame siit teada, et täpseima klassifitseerimistulemuse annab milline k väärtus? 
# Teeme tulemuse põhjal klassifitseerimisprotsessi näitlikult lõpuni läbi ja treenime lõpliku mudeli (kui peame optimaalseks k väärtuseks midagi muud kui see, millel on väikseim mmce väärtus, saab selle järgnevas käsus kirja panna nt k = 6 puhul par.vals = list(k = 6)).

TunedKNN <- setHyperPars(makeLearner("classif.knn"), par.vals = OptimKNN$x)

TunedKnnModel <- train(TunedKNN, spamTask_valid)
TunedKnnModel

# objektis TunedKnnModel on mudel, mida saaksime edaspidi kasutada uutel andmetel, kus klassikuuluvuse tunnust ei ole, st päris andmetel, kus on andmed emailide kohta ilma teabeta, kas tegu on spämmiga või mitte. Selleks saab kasutada juba eelnevalt kasutatud funktsiooni predict, kus argumendile newdata omistatakse uus andmestik:

knnProgn <- predict(TunedKnnModel, newdata = ...)

# tulemus ei olnud küll väga täpne, aga korraliku spämmidetektori loomine nõuab ka palju rohkem erinevaid andmeid. Üks põhjus, miks mudel jäi kehvakeseks, on nn curse of dimensionality. Sellest lähtuvalt on üks võimalus loodud mudelit edasi optimeerida, muutes selle efektiivsust, jättes mudelisse tunnused, mis klassifitseerimise täpsusele n-ö kõige rohkem juurde annavad.

#### Iseseisev ülesanne ####

# Kasutage paketist MASS Bostoni linnastu andmeid, kus on erinevate linnade/linnaosade kohta kinnisvara ja elukeskkonda puudutavad andmed. 

library(MASS)
df <- Boston

# See on näidisandmestik, mida on kasutatud erinevates masinõppe ülesannetes, enamasti selleks, et prognoosida kinnisvara hinda. Käesolevas ülesandes tuleb koostada k lähima naabri meetodil mudel, mis prognoosiks võimalikult täpselt kuritegevuse taset - täpsemalt seda, kas kuritegevuse määr elaniku kohta (tunnus crim) on linnas/linnaosas suurem kui 1 või alla selle. 

# Muuhulgas 

## kontrollige, kas andmeid tuleks standardiseerida, 
## valideerige mudel ristvalideerimisega (kirjeldage ka valideerimisel valitud seadeid), 
## hinnake mudeli täpsust erinevate k väärtuste korral ja leidke selle alusel k optimaalne väärtus.