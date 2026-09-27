# ArchiveTrust HTR
## Från evidensbaserad OCR-telemetri till ett oberoende svenskt HTR-benchmark

**Projektrapport, september 2026**  
**Författare:** Dennis Jensen  
**Repository:** `hypergeek-dev/ArchiveTrust_HCR`

## Innehåll

1. [Sammanfattning](#1-sammanfattning)
2. [Bakgrund och idé](#2-bakgrund-och-idé)
3. [Steg 1: evidens och telemetri](#3-steg-1-evidens-och-telemetri)
4. [Steg 2: övergången till HTR](#4-steg-2-övergången-till-htr)
5. [Steg 3: träning av en svensk Loghi-modell](#5-steg-3-träning-av-en-svensk-loghi-modell)
6. [Steg 4: oberoende benchmark](#6-steg-4-oberoende-benchmark)
7. [Steg 5: analys av skillnaden](#7-steg-5-analys-av-skillnaden)
8. [Slutsatser](#8-slutsatser)
9. [Begränsningar](#9-begränsningar)
10. [Tack och attribution](#10-tack-och-attribution)

## 1. Sammanfattning

ArchiveTrust började inte som ett försök att slå Riksarkivet. Frågan blev i stället:

> Hur långt kan en privat, hobbytränad HTR-modell från grunden komma jämfört med en institutionell modell från Riksarkivet, om båda testas på samma frysta material?

HTR, Historical Handwritten Text Recognition, är ungefär OCR för handskrift.

Den slutliga ArchiveTrust-modellen tränades från grunden med Loghi-HTR på cirka **562 000 svenska handskrivna textrader**. På den interna valideringsmängden nådde modellen **9,98% Character Error Rate (CER)**. Det var ett bra internt resultat, men valideringsmaterialet låg nära träningsmaterialet.

Därför byggdes ett separat benchmark från material från Svea Hovrätt. Materialet granskades och rensades innan någon modell kördes. En första fryst version innehöll sannolikt okorrigerad Transkribus-text och ersattes därför, utan att skrivas över, av en renare version med **4 627 textrader på 105 sidor**.

Resultatet blev:

| Oberoende benchmark | ArchiveTrust Loghi | Riksarkivet Lion |
|---|---:|---:|
| CER | **32,70%** | **18,42%** |
| WER | **67,53%** | **43,63%** |
| Tomma radresultat | 71 | 0 |

Lion generaliserade betydligt bättre på det nya materialet.

Detta betraktas inte som ett misslyckande. Projektets mål var att mäta avståndet på ett ärligt sätt, inte att vinna.

Efter benchmarken undersöktes varför skillnaden var så stor. ArchiveTrust-modellen producerade systematiskt för kort text och hade särskilt stora problem med korta och smala textrader. Beam search förklarade nästan ingenting. Ett test med extra bildbredd gjorde resultatet något sämre. Ett verkligt batchberoende fel i Loghi-inferensen upptäcktes, men det verkar vara en mindre reproducerbarhetsfråga och inte en förklaring till den stora skillnaden mot Lion.

Den återstående skillnaden är inte kausalt uppdelad. Skillnader i förträning, arkitektur, träningsdatans variation och modellkapacitet är rimliga förklaringar, men de testades inte med de ablationsexperiment som hade krävts för att bevisa det.

## 2. Bakgrund och idé

ArchiveTrust började som ett bredare försök att göra OCR- och AI-resultat mer observerbara, spårbara och granskningsbara.

Ett viktigt bidrag från **Anders Hast** hjälpte projektet att få en mer konkret forskningsinriktning mot Historical Handwritten Text Recognition. Hans input bidrog till förflyttningen från ett generellt OCR-system för tillit och telemetri till ett fokuserat svenskt HTR-projekt med mätbara experiment och oberoende utvärdering.

Anders levererade också det Svea Hovrätt-material som till slut användes i projektets oberoende benchmark.

## 3. Steg 1: evidens och telemetri

ArchiveTrust byggdes först kring principen att ett OCR-resultat inte automatiskt är sanning.

Systemet separerade bland annat:

- rå modellutdata;
- normaliserad text;
- mänskliga rättelser;
- modell- och versionsidentitet;
- input och hashvärden;
- fel och tomma resultat;
- experiment och jämförelser.

Det gjorde att misslyckade försök kunde sparas som evidens i stället för att försvinna.

## 4. Steg 2: övergången till HTR

Projektet testade tidigt olika HTR-vägar, bland annat SATRN, en Florence-2-baserad pipeline och Transkribus-relaterade arbetsflöden.

Ett viktigt designbeslut var att hålla **segmentering** och **igenkänning** åtskilda. Två modeller som får olika beskärningar av en rad kan inte jämföras rättvist som rena recognizers.

De tidiga experimenten var små och användes främst för att verifiera metodiken.

Därefter ändrades målet: Dennis skulle inte bara jämföra modeller, utan själv träna en svensk Loghi-modell.

## 5. Steg 3: träning av en svensk Loghi-modell

Träningsmaterialet omfattade ungefär **562 123 textrader från 11 arkivsamlingar**. Korpusen byggdes från Riksarkivets offentligt publicerade dataset i [Training data for Swedish Lion Libre](https://huggingface.co/collections/Riksarkivet/training-data-for-swedish-lion-libre), den publika samling som Riksarkivet använde vid utvecklingen av Swedish Lion Libre.

### Experiment 0

Den första fullskaliga finjusteringen delade arbetet i 57 separata containerkörningar. Optimizer och learning-rate-schema startades om för varje del.

Det visade sig att orkestreringen hade förändrat själva experimentet.

Resultat:

- CER **0,1728**
- WER **0,4945**

### Experiment 1

Nästa körning gjordes som kontinuerlig träning så att optimizer och learning-rate-schema kunde leva vidare genom hela epoken.

Efter fem epoker:

- CER **0,1310**
- WER **0,4027**

### Experiment 2

Efter att dokumentationen lästs om valdes en ny riktning: träna en rekommenderad Loghi-arkitektur **från grunden** på det svenska materialet.

| Epoch | CER | WER |
|---|---:|---:|
| 1 | 0,1918 | 0,5470 |
| 2 | 0,1487 | 0,4546 |
| 3 | 0,1257 | 0,3915 |
| 4 | 0,1145 | 0,3674 |
| 5 | 0,1075 | 0,3524 |
| 6 | 0,1011 | 0,3360 |
| 7 | **0,0998** | **0,3273** |

Epoch 7 har en dokumenterad metodisk reservation. Den långa kontinuerliga körningen kraschade efter epoch 6 och den sista epoken fortsatte från verifierade vikter med ny optimizer och omstartat learning-rate-schema.

Checkpointen är verklig och verifierad, men epoch 7 är därför inte exakt samma typ av kontinuerlig träning som epoch 1-6.

## 6. Steg 4: oberoende benchmark

**Anders Hast** levererade ett Svea Hovrätt-material som senare användes för slutjämförelsen.

Den första kandidatversionen hade **6 486 rader**. En deterministisk completeness-audit hittade indikationer på att en del av texten fortfarande var okorrigerad Transkribus-recognition.

Efter en fast bildgranskning exkluderades **35 hela sidor / 1 859 rader** innan modellerna kördes.

Slutversionen innehöll:

- 4 dokument
- 105 sidor
- 4 627 rader
- 166 201 referenstecken
- 28 872 referensord

Benchmarken frystes innan modellresultat fanns.

Primära inställningar låstes i förväg:

- ArchiveTrust Loghi: beam 10
- Riksarkivet Lion Libre: 4-beam `generation_config`

Resultatet:

| Mått | ArchiveTrust Loghi | Lion |
|---|---:|---:|
| CER | **32,70%** | **18,42%** |
| WER | **67,53%** | **43,63%** |
| Deletions | 32 506 | 9 468 |
| Tomma outputs | 71 | 0 |

Lion var bättre på samtliga fyra dokument och alla 105 sidor.

Det viktiga är inte att Lion vann. Det viktiga är att en hobbytränad scratch-modell nu hade ett ärligt, reproducerbart avstånd till en etablerad institutionell modell.

## 7. Steg 5: analys av skillnaden

Efter benchmarken gjordes ingen omedelbar reträning.

Först undersöktes om gapet berodde på ett enkelt inferens- eller preprocessingproblem.

### Underproduktion

Loghi producerade generellt för lite text.

Genomsnittlig outputlängd jämfört med referens:

- Loghi: **0,80**
- Lion: **0,98**

### Korta och smala rader

De 71 tomma Loghi-resultaten kom nästan helt från extremt korta/smala beskärningar.

En hypotes var därför att CTC-nätverket fick för få tidssteg.

### Beam search

Beam 1 gav CER **33,00%**, beam 10 gav **32,70%**.

Skillnaden var bara 0,30 procentenheter. Beam search förklarar alltså nästan inget av gapet.

### Paddingexperiment

Ett förregistrerat test ökade smala crops till 230 px minsta bredd.

Resultatet blev sämre:

- CER **32,70% -> 32,72%**
- tomma resultat **71 -> 138**

Hypotesen motsades av experimentet.

### Batchberoende

Ett riktigt reproducerbarhetsproblem hittades i Loghi: samma oförändrade crop kunde få olika prediction beroende på vilka andra bilder den delade batch med.

Det bekräftades deterministiskt i kontrollerade tester.

Effekten var dock normalt liten och förklarade varken de 71 tomma resultaten eller en meningsfull del av skillnaden mot Lion.

### Försök med riktig sequence length

En patch gav varje sample sin egen CTC sequence length. Patchen fungerade korrekt i isolerade enhetstester, men batchberoendet fanns kvar på riktiga crops.

En förregistrerad stopping rule sa att full benchmark inte skulle köras om mekanismen inte försvann.

Resultat: **MECHANISM INCOMPLETE**.

Den officiella CER-siffran **32,70%** ändrades därför aldrig.

## 8. Slutsatser

Projektet visar framför allt fyra saker:

1. Ett starkt internt valideringsresultat är inte samma sak som extern generalisering.
2. En privat scratch-modell kan komma långt nog för att göra en seriös jämförelse men låg fortfarande tydligt efter Lion på det oberoende materialet.
3. Tekniska pipelinefel kan hittas genom evidens och reproducerbara diagnostikexperiment, men alla fel förklarar inte huvudresultatet.
4. Negativa experiment är användbara resultat när de gör att en rimlig hypotes kan avskrivas.

Det återstående gapet är förenligt med skillnader i bland annat förträning, arkitektur, modellkapacitet och variation i träningsdata, men projektet bevisade inte hur stor andel varje faktor står för.

## 9. Begränsningar

- Referenstranskriptionen är studentproducerad och inte certifierad professionell ground truth.
- Källmaterialet hade fortfarande Transkribus-status `IN_PROGRESS`.
- En sannolikt okorrigerad del togs bort före benchmarken, men vanliga mänskliga transkriptionsfel kan finnas kvar.
- Träningsöverlapp kunde bara kontrolleras mot den kända Svea Hovrätt-delen av träningsmaterialet.
- Lion och Loghi skiljer sig samtidigt i arkitektur, förträning, träningsdata och preprocessing.
- Experiment 2 epoch 7 har en optimizer/LR-kontinuitetsreservation.
- Det kvarvarande batchberoendet är inte fullständigt förklarat.
- Ingen post-benchmark reträning gjordes.

## 10. Tack och attribution

### Anders Hast

**Anders Hast, He/Him**  
Professor in Computerised Image Processing at Uppsala University  
Distinguished University Teacher, InfraVis Faculty  
LinkedIn: <https://www.linkedin.com/in/anders-hast-15536372/>

Ett särskilt tack till Anders Hast för hans bidrag till projektets forskningsinriktning och för att han levererade datamaterialet som användes i det slutliga oberoende benchmark-testet. Hans input var en viktig del i utvecklingen från ett generellt OCR-system för tillit och telemetri till ett fokuserat svenskt HTR-projekt.

### Riksarkivet

ArchiveTrusts träningskorpus byggdes från Riksarkivets offentligt publicerade historiska handskriftsdataset i [Training data for Swedish Lion Libre](https://huggingface.co/collections/Riksarkivet/training-data-for-swedish-lion-libre).

Ett särskilt tack till Riksarkivet för att detta träningsmaterial gjorts offentligt tillgängligt. Dataseten gjorde det möjligt att genomföra projektets svenska scratch-träning i en omfattning som annars hade varit svår att uppnå i ett fristående projekt.

---

ArchiveTrust blev därmed inte en berättelse om att en hobbyutvecklare slog en myndighetsmodell.

Det blev en berättelse om att bygga tillräckligt mycket metodik för att kunna mäta avståndet på riktigt.
