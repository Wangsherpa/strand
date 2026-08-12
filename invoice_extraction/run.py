"""Run the Invoice Extraction Workflow against test emails."""

import sys
from pathlib import Path

# sys.path.append("/Users/wangsherpa/Desktop/genai/projects")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# sys.path.append("/Users/wangsherpa/Desktop/genai/projects/strand")

from invoice_extraction.workflow import InvoiceExtractionWorkflow

# ---------------------------------------------------------------------------
# Test emails
# ---------------------------------------------------------------------------

INVOICE_EMAIL = {
    "subject": "Invoice #INV-2026-0781 from Acme Cloud Services — Due Aug 15",
    "sender": "billing@acmecloud.com",
    "body": (
        "Dear Procurement Team,\n\n"
        "Please find attached your monthly invoice for cloud infrastructure "
        "services rendered in July 2026.\n\n"
        "Invoice Number: INV-2026-0781\n"
        "Date Issued: July 26, 2026\n"
        "Due Date: August 15, 2026\n\n"
        "Summary of Charges:\n"
        "  - Compute Instances (t3.large x 12):      $4,320.00\n"
        "  - Managed Database (RDS PostgreSQL):       $1,850.00\n"
        "  - Object Storage (S3, 14 TB):              $1,120.00\n"
        "  - Content Delivery Network:                  $640.00\n"
        "  - Technical Support (Business tier):         $500.00\n"
        "  -----------------------------------------------\n"
        "  Subtotal:                                  $8,430.00\n"
        "  Sales Tax (8.25%):                           $695.48\n"
        "  Total Due:                                 $9,125.48\n\n"
        "Seller: Acme Cloud Services Inc.\n"
        "Buyer:  Wang Inc.\n"
        "Payment Terms: Net 30\n"
        "Payment Method: ACH Transfer\n\n"
        "For questions, contact billing@acmecloud.com or call (888) 555-0199.\n\n"
        "Thank you for your business.\n"
        "— Acme Cloud Services Billing Department"
    ),
}

NON_INVOICE_EMAIL = {
    "subject": "Re: Q3 product roadmap feedback — need your input by Friday",
    "sender": "sherpa@example.com",
    "body": (
        "Hey team,\n\n"
        "Quick reminder — I need everyone's feedback on the Q3 roadmap "
        "proposal by this Friday EOD. The draft is in the shared drive "
        "under /Product/Roadmap/2026-Q3/.\n\n"
        "Key areas I'd like feedback on:\n"
        "1. Analytics dashboard v2 — should we prioritize real-time or batch?\n"
        "2. User permissions overhaul — timeline realistic?\n"
        "3. API rate limiting — do we need it this quarter?\n\n"
        "Please add comments directly in the doc. We'll discuss in Monday's "
        "standup.\n\n"
        "Thanks,\n"
        "Sherpa\n"
        "VP Product, Wang Inc."
    ),
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    wf = InvoiceExtractionWorkflow()

    # --- Invoice email ---
    print("=" * 60)
    print("EMAIL #1 — Invoice")
    print("=" * 60)
    print(f"  Subject: {INVOICE_EMAIL['subject']}")
    print()

    result = wf.run(INVOICE_EMAIL)

    classification = result.nodes.get("classify")
    print(f"  Classification:")
    print(f"    is_invoice: {classification.is_invoice}")
    print(f"    confidence: {classification.confidence}")
    print(f"    reasoning:  {classification.reasoning}")

    if "extract" in result.nodes:
        e = result.nodes["extract"]
        print(f"\n  Extraction:")
        print(f"    invoice_number: {e.invoice_number}")
        print(f"    amount:         {e.amount}")
        print(f"    currency:       {e.currency}")
        print(f"    seller:         {e.seller}")
        print(f"    buyer:          {e.buyer}")
        print(f"    product:        {e.product}")
        print(f"    due_date:       {e.due_date}")
        print(f"    summary:        {e.summary}")
    elif "not_invoice" in result.nodes:
        print(f"\n  {result.nodes['not_invoice'].result}")

    # --- Non-invoice email ---
    print()
    print("=" * 60)
    print("EMAIL #2 — Product Roadmap (non-invoice)")
    print("=" * 60)
    print(f"  Subject: {NON_INVOICE_EMAIL['subject']}")
    print()

    result = wf.run(NON_INVOICE_EMAIL)

    classification = result.nodes.get("classify")
    print(f"  Classification:")
    print(f"    is_invoice: {classification.is_invoice}")
    print(f"    confidence: {classification.confidence}")
    print(f"    reasoning:  {classification.reasoning}")

    not_invoice = result.nodes.get("not_invoice")
    if not_invoice:
        print(f"\n  Result: {not_invoice.result}")

    print()
    print("✅ Done.")
