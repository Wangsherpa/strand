"""Event schema for the invoice extraction workflow."""

from pydantic import BaseModel


class EmailEvent(BaseModel):
    subject: str
    body: str
    sender: str
