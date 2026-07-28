"""The frozen task ladder, ordered by expected fragility.

Prompts and fixtures are immutable once the pre-registration is frozen.
All fixtures are synthetic and labeled as such in their own text — no real
company, person, or client data. Prompts are pure ASCII so canonical
serialization has no encoding wrinkles.
"""

_AGREEMENT_FIXTURE = (
    "SERVICES AGREEMENT (EXCERPT - SYNTHETIC TEST FIXTURE) "
    "This Services Agreement is entered into as of March 3, 2025 (the "
    '"Effective Date") by and between Halvern Logistics LLC ("Client") and '
    'Corvid Systems Group Inc. ("Provider"). Provider will supply warehouse '
    "telemetry integration services as described in Statement of Work No. 4. "
    "The parties acknowledge purchase order PO-83614-QN governs all invoicing "
    "under this Agreement. Monthly service fees are $12,740, invoiced net-30. "
    "The initial term is twenty-four (24) months from the Effective Date, "
    "renewing automatically for successive twelve (12) month periods unless "
    "either party provides ninety (90) days written notice. This Agreement is "
    "governed by the laws of the State of Illinois. Provider's liability cap "
    "is limited to fees paid in the twelve (12) months preceding any claim. "
    "Support requests are acknowledged within four (4) business hours."
)

_TICKET_FIXTURE = (
    "Hi - I was charged twice this month after the mobile app crashed during "
    "checkout on Tuesday. The duplicate charge shows as pending. The app "
    "still crashes every time I open my order history, so I can't even "
    "confirm what went through. Can someone look into this?"
)

_INVENTORY_FIXTURE = (
    "Inventory note (synthetic): Item Corvid CS-220 badge printer, SKU "
    "CS220-BDG-K, currently 17 units on hand at $349.50 per unit, located in "
    "aisle 9 bin 4, reorder threshold 6 units, supplier Corvid Systems Group, "
    "last audited 2025-02-11, item is active for sale."
)

TASKS = {
    "extraction": {
        "description": (
            "Pull one named field from a fixed ~200-token document. Short, "
            "constrained output; expected most reproducible."
        ),
        "prompt": (
            "Read the following contract excerpt and output only the "
            "purchase order number exactly as it appears, with no other "
            "text.\n\n" + _AGREEMENT_FIXTURE
        ),
    },
    "classification": {
        "description": (
            "Single label from a closed set; output is one token. The ticket "
            "deliberately straddles BILLING and TECHNICAL so the task probes "
            "argmax flipping near a decision boundary rather than a trivially "
            "dominant logit."
        ),
        "prompt": (
            "Classify the following support ticket into exactly one of these "
            "categories: BILLING, TECHNICAL, ACCOUNT, GENERAL. Output only "
            "the single category word, with no other text.\n\nTicket: "
            + _TICKET_FIXTURE
        ),
    },
    "structured_json": {
        "description": (
            "Populate a fixed schema from a fixture; the production-realistic "
            "case."
        ),
        "prompt": (
            "From the inventory note below, output only a JSON object with "
            "exactly these keys: sku, name, quantity_on_hand, unit_price_usd, "
            "reorder_threshold, in_stock. Use a number for quantity_on_hand, "
            "unit_price_usd, and reorder_threshold, and a boolean for "
            "in_stock. No markdown fences, no commentary.\n\n"
            + _INVENTORY_FIXTURE
        ),
    },
    "open_generation": {
        "description": (
            "About 400 words of prose; the longest divergence surface and "
            "expected least reproducible."
        ),
        "prompt": (
            "Explain in about 400 words why floating-point addition is not "
            "associative on modern hardware, and how this affects the "
            "reproducibility of large matrix multiplications executed with "
            "different reduction orders. Write plain prose with no headings, "
            "no lists, and no code."
        ),
    },
}

# --- Study 2 Q4: sparse input-length ladder (prereg v2) --------------------
# Deterministic context padding prepended to the extraction task. Char-exact
# targets approximate 1k / 10k / 50k input tokens at typical English
# tokenization (~3.7 chars per token); the exact token count is irrelevant —
# the ladder is sparse, and the padded prompts are byte-identical within a
# cell and across planes like every other fixture. Pure ASCII.

_PAD_PARAGRAPH = (
    "Background operations log (synthetic filler, not relevant to the task): "
    "the warehouse floor completed its scheduled cycle count without "
    "discrepancies, conveyor line three resumed after routine belt "
    "maintenance, inbound dock assignments rotated per the standard weekly "
    "plan, and the facilities team recorded nominal temperature and humidity "
    "readings across all storage zones throughout the reporting period. "
)

PAD_CHAR_TARGETS = {"1k": 3_700, "10k": 37_000, "50k": 185_000}


def padded_prompt(pad_label, base_prompt):
    """base_prompt preceded by char-exact deterministic filler."""
    target = PAD_CHAR_TARGETS[pad_label]
    reps = -(-target // len(_PAD_PARAGRAPH))
    filler = (_PAD_PARAGRAPH * reps)[:target]
    return filler + "\n\n" + base_prompt
