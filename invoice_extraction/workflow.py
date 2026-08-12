"""Invoice Extraction Workflow — classifies and extracts invoice details from emails."""

from strand.core import NodeConfig, Workflow, WorkflowSchema

from invoice_extraction.nodes import (
    ClassifyNode,
    ExtractNode,
    InvoiceRouter,
    NotInvoiceNode,
)
from invoice_extraction.schema import EmailEvent


class InvoiceExtractionWorkflow(Workflow):
    workflow_schema = WorkflowSchema(
        description=(
            "Classifies incoming emails as invoice-related or not. "
            "If invoice-related, extracts amount, currency, product, seller, buyer, "
            "invoice number, and due date."
        ),
        event_schema=EmailEvent,
        start="classify",
        nodes=[
            NodeConfig(
                node="classify",
                connections=["invoice_router"],
                description="LLM: classify whether email contains an invoice",
            ),
            NodeConfig(
                node="invoice_router",
                connections=["extract", "not_invoice"],
                is_router=True,
                description="Route to extract or not_invoice based on classification",
            ),
            NodeConfig(
                node="extract",
                description="LLM: extract amount, currency, product, seller, buyer, dates",
            ),
            NodeConfig(
                node="not_invoice",
                description="Stop — email is not invoice-related",
            ),
        ],
        registry={
            "classify":       ClassifyNode,
            "invoice_router": InvoiceRouter,
            "extract":        ExtractNode,
            "not_invoice":    NotInvoiceNode,
        },
    )
