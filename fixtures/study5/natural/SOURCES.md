# Study 5 natural arm — sources, licenses, provenance

Both feeds are public-domain government open data. Snapshots are committed in `snapshots/`
so the draw (PROTOCOL §7.2) is re-derivable by anyone; the feeds themselves update daily, so
the committed bytes — not the live feed — are the reference. Verified 2026-09-09.

## 1. City of Austin, Texas — Arterial Management Materials Warehouse Inventory

- Dataset: https://data.austintexas.gov/Transportation-and-Mobility/Arterial-Management-Materials-Warehouse-Inventory/hcaw-evi2
  (Socrata id `hcaw-evi2`; SODA endpoint `https://data.austintexas.gov/resource/hcaw-evi2.json`)
- Publisher / attribution: City of Austin, Texas — data.austintexas.gov
- Description (dataset metadata): "This dataset contains information about the materials
  warehouse for Austin Transportation and Public Works Arterial Management Division. It is
  updated daily and provides a running count of materials on-hand, and is used for internal
  reporting purposes."
- License: the dataset record carries no license field; the City of Austin Open Data Terms of
  Use (https://data.austintexas.gov/d/2z87-8uh7, PDF) state, under "Licensing Data":
  > "COA data available through data.austintexas.gov and the City of Austin's ArcGIS Downloads
  > Page is offered free and without restriction. Data and content created by COA government
  > employees within the scope of their employment are not subject to copyright protection."
  and describe both portals as "public domain websites". (The PDF's embedded font is
  glyph-remapped; the quote was recovered by known-plaintext decoding with zero mapping
  conflicts and should be re-read from the PDF by anyone relying on it.)
- Snapshot: `snapshots/austin-hcaw-evi2.json` — the feed's latest `published_date`
  (`2026-09-08T23:13:18.000`; the feed appends a dated copy of the whole inventory daily, 1,675 dates,
  521,157 rows total at fetch), SODA query `$where=published_date='…'&$order=id`,
  fetched 2026-09-10T03:33:10Z, 381 rows, sha256 `01220c3ac0393124979ebeee657f53485b57f0a20a1f976d2aed43e4e32d7aad`.
- Columns used: `financial_name`, `common_name`, `tracking_type`, `stock_number`, `category`,
  `object`, `unit_cost`, `unit_of_measure`, `status`, `re_order_threshold`, `total_on_hand`,
  `total_value`, `re_order_turnaround_time`, `re_order_status` (rendered with their Socrata
  display names); `id`, `published_date`, `modified_date` are record-keeping and excluded from
  the rendered document (kept in each item's `source.raw`).

## 2. Montgomery County, Maryland — ABS Store Inventory and Sale Items

- Dataset: https://data.montgomerycountymd.gov/d/ib5t-5ncy (Socrata id `ib5t-5ncy`; SODA
  endpoint `https://data.montgomerycountymd.gov/resource/ib5t-5ncy.json`); catalog entry
  https://catalog.data.gov/dataset/abs-store-inventory-and-sale-items
- Publisher / attribution: Montgomery County, MD (dataMontgomery)
- Description (dataset metadata): "This dataset provide a listing of inventory items, including
  store quantities and sale prices. Update Frequency : Daily"
- License: dataset metadata `license: "Public Domain"` (`licenseId: PUBLIC_DOMAIN`), read from
  `https://data.montgomerycountymd.gov/api/views/ib5t-5ncy.json` on 2026-09-09. Portal terms:
  https://data.montgomerycountymd.gov/terms-of-use
- Snapshot: `snapshots/montgomery-ib5t-5ncy.json` — the live view, SODA query `$order=code`,
  fetched 2026-09-10T03:33:10Z (feed `rowsUpdatedAt` 1788955229 = 2026-09-09T12:00:29Z), 7037 rows,
  sha256 `a632cd5b0b0f83c125e6858b5198c406357ae6e5d084defe87bc992a3daef55b`.
- Columns (all rendered with their Socrata display names): `code` Item Code, `category` Item
  Category, `description` Item Description, `size` Item Size, `totalinventory` Total Amount in
  Inventory, `price` Item Price, `saleprice` Item Sale Price, `saleenddate` Sale End Date.

## 3. What was NOT done

- No row was hand-picked, reworded, corrected, or dropped after the draw. Eligibility, seed,
  draw, and rendering are fixed in `harness/study5_natural.py`; `python3 -m
  harness.study5_natural check` re-derives the corpus from the snapshots.
- No model saw any item before labeling (the chain scripts refuse on an unfrozen corpus; the
  labels were assigned from the rendered text alone).
- No personal data: both feeds are product/material inventories.
