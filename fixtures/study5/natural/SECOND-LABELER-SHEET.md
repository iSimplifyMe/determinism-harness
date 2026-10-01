# Study 5 natural arm — second-labeler sheet (20 items)

Label each record from the text below ONLY. Do not open `natural_corpus.json` or
PROTOCOL.md section 7 until you have finished — your labels are the independent
check on the first labeler's conventions, and reading them first would defeat it.

For each item fill in:

- `item_name`, `unit_price` (number only, no `$`), `quantity_in_stock` (integer only) —
  what a careful cataloger would report for this record; `null` where the record is silent.
- `class`: `clean` (one reading) / `near_tie` (one correct reading, but a token invites
  a specific misread) / `ambiguous` (two readings an expert could defend — list both).
- Two readings: separate them with ` | ` (space, pipe, space), your primary FIRST,
  e.g. `- item_name: ACME WIDGET 750ML | ACME WIDGET`. One value per line otherwise.
- `note`: one line, optional.

Fill the lines in place and save this file; say "sheet done" when finished.

Return the completed sheet (or just the filled blocks) — agreement is computed at
freeze on the three field values and the class.

## s5n-001

```
Financial Name: POLES, TRAFFIC SIGNAL, STEEL TYPE 2W
Common Name: Pole, Signal, Galvanized, Type 2W
Tracking Type: FINANCIAL
Stock Number: 55085590003
Category: Poles
Object: 7148 |  Poles - towers - steel
Unit Cost: 6719.5152
Unit of Measure: EA
Status: ACTIVE
Re-Order Threshold: 2
Total On Hand: 9
Total Value: 60475.6368
Re-Order Turnaround Time: 8
Re-Order Status: Not Needed
```

- item_name:Pole, Signal, Galvanized, Type 2W
- unit_price:6719.5152
- quantity_in_stock:9
- class:near_tie
- note: category=Poles

## s5n-002

```
Financial Name: 100 Amp Breaker Panel
Common Name: Breaker Panel, 100 AMP
Tracking Type: FINANCIAL
Stock Number: 55089570002
Category: Misc. Items
Object: 7127 |  Electrical/lighting
Unit Cost: 123.2989
Unit of Measure: EA
Status: ACTIVE
Re-Order Threshold: 5
Total On Hand: 8
Total Value: 986.3912
Re-Order Turnaround Time: 2
Re-Order Status: Not Needed
```

- item_name:Breaker Panel, 100 AMP
- unit_price:123.2989
- quantity_in_stock:8
- class:near_tie
- note: category=Misc. Items

## s5n-003

```
Financial Name: School Beacon Cabinet Time Clock/Switch & Sensor
Common Name: School Beacon Cabinet Time Clock/Switch
Tracking Type: FINANCIAL
Stock Number: 55088770015
Category: Flashers
Object: 7122 |  Hardware/wire/steel
Unit Cost: 799
Unit of Measure: EA
Status: ACTIVE
Re-Order Threshold: 10
Total On Hand: 14
Total Value: 11186
Re-Order Turnaround Time: 4
Re-Order Status: Not Needed
```

- item_name:School Beacon Cabinet Time Clock/Switch
- unit_price:799
- quantity_in_stock:14
- class:near_tie
- note: category=Flashers

## s5n-004

```
Financial Name: CCTV Camera Pendant Mount 1.5 NPT, 120" Galvanized Cable
Common Name: CCTV Camera Pendant Mount 1.5 NPT, 120" Galvanized Cable
Tracking Type: FINANCIAL
Stock Number: 84084280048
Category: Communications & Monitoring
Object: 7122 |  Hardware/wire/steel
Unit Cost: 250.7273
Unit of Measure: EA
Status: ACTIVE
Re-Order Threshold: 5
Total On Hand: 7
Total Value: 1755.0911
Re-Order Turnaround Time: 3
Re-Order Status: Not Needed
```

- item_name:CCTV Camera Pendant Mount 1.5 NPT, 120" Galvanized Cable
- unit_price:250.7273
- quantity_in_stock:7
- class: clean
- note: category=Communications & Monitoring

## s5n-005

```
Financial Name: CLIPS BANDING SS 3/4 IN
Common Name: "Stainless Steel Wing Clips - 100 per box. (Width 3/4'')"
Tracking Type: FINANCIAL
Stock Number: 32075371001
Category: Banding Equipment
Object: 7127 |  Electrical/lighting
Unit Cost: 82.9499
Unit of Measure: EA
Status: ACTIVE
Re-Order Threshold: 10
Total On Hand: 15
Total Value: 1244.2485
Re-Order Turnaround Time: 2
Re-Order Status: Not Needed
```

- item_name:CLIPS BANDING SS 3/4 IN
- unit_price:82.9499
- quantity_in_stock:15
- class: ambiguous
- note: category=Banding Equipment

## s5n-006

```
Financial Name: Etherwan Switch Power Supply
Common Name: Etherwan Switch Power Supply
Tracking Type: FINANCIAL
Stock Number: 20621750012
Category: Ethernet Switches & Cradlepoint Equipment
Object: 7122 |  Hardware/wire/steel
Unit Cost: 64
Unit of Measure: EA
Status: ACTIVE
Re-Order Threshold: 0
Total On Hand: 0
Total Value: 0
Re-Order Status: Not Needed
```

- item_name:Etherwan Switch Power Supply
- unit_price:64
- quantity_in_stock:0
- class: clean
- note: category=Ethernet Switches & Cradlepoint Equipment

## s5n-007

```
Financial Name: Pedestrian Button Base, Polara Bulldog
Common Name: Pedestrian Button Base, Polara Bulldog
Tracking Type: FINANCIAL
Stock Number: 55089620007
Category: Pedestrian Buttons and Beepers
Object: 7127 |  Electrical/lighting
Unit Cost: 16.969
Unit of Measure: EA
Status: ACTIVE
Re-Order Threshold: 50
Total On Hand: 23
Total Value: 390.287
Re-Order Turnaround Time: 3
Re-Order Status: Re-Order Needed
```

- item_name:Pedestrian Button Base, Polara Bulldog
- unit_price:16.969
- quantity_in_stock:23
- class: clean
- note: category=Pedestrian Buttons and Beepers

## s5n-008

```
Financial Name: Detection, Currux - 4 Approach
Common Name: Detection, Currux - 4 Approach
Tracking Type: FINANCIAL
Stock Number: 55091780016
Category: Video Detection Equipment
Object: 7122 |  Hardware/wire/steel
Unit Cost: 15015
Unit of Measure: EA
Status: ACTIVE
Re-Order Threshold: 5
Total On Hand: 15
Total Value: 225225
Re-Order Turnaround Time: 3
Re-Order Status: Not Needed
```

- item_name:Detection, Currux - 4 Approach
- unit_price:15015
- quantity_in_stock:15
- class:clean
- note: category=Video Detection Equipment

## s5n-009

```
Financial Name: Tomar Dual Channel Preemption Card
Common Name: Tomar Dual Channel Preemption Card
Tracking Type: FINANCIAL
Stock Number: 55081270011
Category: Preemption
Object: 7122 |  Hardware/wire/steel
Unit Cost: 1526.179998
Unit of Measure: EA
Status: ACTIVE
Re-Order Threshold: 0
Total On Hand: 0
Total Value: 0
Re-Order Status: Not Needed
```

- item_name:Tomar Dual Channel Preemption Card
- unit_price:1526.179998
- quantity_in_stock:0
- class:clean
- note: category=Preemption

## s5n-010

```
Financial Name: LED Red Arrow with Molex connectors
Common Name: LED Red Arrow with Molex connectors
Tracking Type: FINANCIAL
Stock Number: 55088440004
Category: LEDs
Object: 7127 |  Electrical/lighting
Unit Cost: 36.1891
Unit of Measure: EA
Status: ACTIVE
Re-Order Threshold: 40
Total On Hand: 483
Total Value: 17479.3353
Re-Order Turnaround Time: 3
Re-Order Status: Not Needed
```

- item_name:LED Red Arrow with Molex connectors
- unit_price:36.1891
- quantity_in_stock:483
- class:clean
- note: category=LEDs

## s5n-026

```
Code: 246323
Category: AMERICAN WHITE
Description: MENAGE A TROIS LIGHT PINOT GRIGIO 750ML
Size: 750ML
Total Inventory: 204
Price: 12.99
```

- item_name:null
- unit_price:12.99
- quantity_in_stock:
- class:ambiguous
- note: category=AMERICAN WHITE

## s5n-027

```
Code: 43931
Category: AMERICAN RED
Description: ST FRANCIS SONOMA CAB VV 750ML
Size: 750ML
Total Inventory: 552
Price: 21.99
```

- item_name:null
- unit_price:21.99
- quantity_in_stock:552
- class:ambiguous
- note: category=AMERICAN RED

## s5n-028

```
Code: 425419
Category: AMERICAN RED
Description: ACROBAT P/NOIR - 750ML
Size: 750ML
Total Inventory: 324
Price: 19.99
```

- item_name:null
- unit_price:19.99
- quantity_in_stock:324
- class:ambiguous
- note: category=AMERICAN RED

## s5n-029

```
Code: 246628
Category: SESSION RTD
Description: SURFSIDE LEMONADE VARIETY PACK 2/12PK 12OZ CAN - 12.0Z (12-Pack)
Size: 12pk
Total Inventory: 40032
Price: 25.99
```

- item_name:null
- unit_price:25.99
- quantity_in_stock:40032
- class:ambiguous
- note: category=SESSION RTD

## s5n-030

```
Code: 79600
Category: ITALIAN RED WINE
Description: DAVINCI CHIANTI DOCG - 750ML
Size: 750ML
Total Inventory: 360
Price: 14.99
```

- item_name:null
- unit_price:14.99
- quantity_in_stock:360
- class:ambiguous
- note: category=ITALIAN RED WINE

## s5n-031

```
Code: 88516
Category: SPARKLING
Description: CHANDON GARDEN SPRITZER - 750ML
Size: 750ML
Total Inventory: 114
Price: 27.99
```

- item_name:null
- unit_price:27.99
- quantity_in_stock:114
- class:ambiguous
- note: category=SPARKLING

## s5n-032

```
Code: 85700
Category: WINE CANS
Description: DARK HORSE ROSE BUBBLES - CAN - 375ML
Size: 375ML
Total Inventory: 264
Price: 5.99
```

- item_name:null
- unit_price:5.99
- quantity_in_stock:264
- class:ambiguous
- note: category=WINE CANS

## s5n-033

```
Code: 87416
Category: SPARKLING
Description: ANDRE WHITE CHAMPAGNE - 750ML
Size: 750ML
Total Inventory: 840
Price: 9.99
```

- item_name:null	
- unit_price:9.99
- quantity_in_stock:840
- class:ambiguous
- note: category=SPARKLING

## s5n-034

```
Code: 355790
Category: ITALIAN WHITE WINE
Description: STELLA ROSA WATERMELON - 750ML
Size: 750ML
Total Inventory: 300
Price: 13.99
```

- item_name:null
- unit_price:13.99
- quantity_in_stock:300
- class:ambiguous
- note: category=ITALIAN WHITE WINE

## s5n-035

```
Code: 50253
Category: IMPORTED CORDIALS
Description: SUNTORY MIDORI MELON - 750ML
Size: 750ML
Total Inventory: 144
Price: 20.99
```

- item_name:null
- unit_price:20.99
- quantity_in_stock:
- class:ambiguous
- note: category=IMPORTED CORDIALS
