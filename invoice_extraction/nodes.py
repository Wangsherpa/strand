"""Nodes for the invoice extraction workflow.

ClassifyNode and ExtractNode use ``strand.llm.LLMNode``.
NotInvoiceNode is pure logic. InvoiceRouter handles branching.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field

from strand.core import BaseRouter, Node, RouterNode, TaskContext
from strand.llm import LLMConfig, LLMNode

# ---------------------------------------------------------------------------
# Bootstrap — load API keys from strand/.env
# ---------------------------------------------------------------------------

_ENV_PATH = Path(__file__).resolve().parent.parent / "strand" / ".env"
load_dotenv(_ENV_PATH)

# ===================================================================
# LLM Nodes
# ===================================================================


class ClassifyNode(LLMNode):
    """Classifies whether an email contains an invoice."""

    class OutputType(LLMNode.OutputType):
        is_invoice: bool = Field(
            description="True if the email contains or references an invoice"
        )
        confidence: str = Field(description="One of: high, medium, low")
        reasoning: str = Field(
            description="One-sentence explanation of the classification"
        )

    def get_llm_config(self) -> LLMConfig:
        return LLMConfig(
            model="gpt-4o-mini",
            system_prompt=(
                "You classify emails to determine whether they contain an INVOICE "
                "(a bill, payment request, receipt with line items, purchase order, "
                "or payment confirmation). "
                "Look for: invoice numbers, line items with prices, billing addresses, "
                "payment terms, due dates, tax amounts, purchase order references. "
                "Be strict — only return is_invoice=True when the email clearly "
                "contains or references a specific invoice or bill."
            ),
        )

    async def build_user_message(self, ctx: TaskContext) -> str:
        return (
            f"Subject: {ctx.event.subject}\n"
            f"Sender:  {ctx.event.sender}\n"
            f"Body:\n{ctx.event.body}"
        )


class ExtractNode(LLMNode):
    """Extracts invoice details from the email."""

    class OutputType(LLMNode.OutputType):
        invoice_number: str = Field(
            description="Invoice or reference number, or 'unknown'"
        )
        amount: str = Field(
            description="Total invoice amount as a string with currency symbol, e.g. '$1,250.00' or 'unknown'"
        )
        currency: str = Field(
            description="ISO 4217 currency code (USD, EUR, GBP, etc.), or 'unknown'"
        )
        seller: str = Field(
            description="Company or person issuing the invoice, or 'unknown'"
        )
        buyer: str = Field(
            description="Company or person being billed, or 'unknown'"
        )
        product: str = Field(
            description="Description of products or services being billed, or 'unknown'"
        )
        due_date: str = Field(
            description="Payment due date in YYYY-MM-DD format, or 'unknown'"
        )
        summary: str = Field(
            description="One-paragraph summary of the invoice contents"
        )

    def get_llm_config(self) -> LLMConfig:
        return LLMConfig(
            model="gpt-4o-mini",
            system_prompt=(
                "You extract structured information from invoice emails. "
                "Extract the following fields precisely. If a field is not "
                "mentioned in the email, set it to 'unknown'.\n\n"
                "- invoice_number: any invoice ID, reference number, or order number\n"
                "- amount: the total amount with currency symbol if present\n"
                "- currency: the ISO 4217 currency code (USD, EUR, GBP, CAD, AUD, etc.)\n"
                "- seller: the vendor, supplier, or company sending the invoice\n"
                "- buyer: the customer, client, or company being billed\n"
                "- product: what was purchased (products, services, subscriptions)\n"
                "- due_date: normalize to YYYY-MM-DD format\n"
                "- summary: a concise one-paragraph summary of the invoice"
            ),
        )

    async def build_user_message(self, ctx: TaskContext) -> str:
        return (
            f"Subject: {ctx.event.subject}\n"
            f"Sender:  {ctx.event.sender}\n"
            f"Body:\n{ctx.event.body}"
        )


# ===================================================================
# Logic Node
# ===================================================================


class NotInvoiceNode(Node):
    """Marks the email as not invoice-related and stops the workflow."""

    class OutputType(Node.OutputType):
        result: str

    async def process(self, ctx: TaskContext) -> TaskContext:
        classify = ctx.nodes.get("classify")
        reason = classify.reasoning if classify else "No classification available."
        self.save_output(
            self.OutputType(
                result=f"This email is NOT invoice-related. Reason: {reason}"
            )
        )
        ctx.stop_workflow()
        return ctx


# ===================================================================
# Router
# ===================================================================


class InvoiceRule(RouterNode):
    """Route to extract if the email contains an invoice."""

    def determine_next_node(self, ctx: TaskContext) -> str | None:
        classify = ctx.nodes.get("classify")
        if classify is not None and classify.is_invoice:
            return "extract"
        return None


class InvoiceRouter(BaseRouter):
    routes = [InvoiceRule()]
    fallback = "not_invoice"
