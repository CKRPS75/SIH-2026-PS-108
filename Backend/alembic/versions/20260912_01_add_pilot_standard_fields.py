"""Add pilot standard ingestion fields."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260912_01"
down_revision = "20260908_01"
branch_labels = None
depends_on = None


json_storage_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("standards", sa.Column("source_dataset", sa.String(255)))
    op.add_column("standards", sa.Column("source_record_id", sa.String(255)))
    op.add_column("standards", sa.Column("source_revision", sa.String(255)))
    op.add_column("standards", sa.Column("source_page_start", sa.Integer()))
    op.add_column("standards", sa.Column("source_page_end", sa.Integer()))
    op.add_column("standards", sa.Column("lifecycle_status", sa.String(64)))
    op.add_column("standards", sa.Column("lifecycle_status_note", sa.Text()))
    op.add_column("standards", sa.Column("retrieval_text", sa.Text()))
    op.add_column("standards", sa.Column("normalized_retrieval_text", sa.Text()))
    op.add_column("standards", sa.Column("full_text", sa.Text()))
    op.add_column(
        "standards",
        sa.Column(
            "search_profile",
            json_storage_type,
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.create_index("ix_standards_source_dataset", "standards", ["source_dataset"])
    op.create_index("ix_standards_source_record_id", "standards", ["source_record_id"])


def downgrade() -> None:
    op.drop_index("ix_standards_source_record_id", table_name="standards")
    op.drop_index("ix_standards_source_dataset", table_name="standards")
    op.drop_column("standards", "search_profile")
    op.drop_column("standards", "full_text")
    op.drop_column("standards", "normalized_retrieval_text")
    op.drop_column("standards", "retrieval_text")
    op.drop_column("standards", "lifecycle_status_note")
    op.drop_column("standards", "lifecycle_status")
    op.drop_column("standards", "source_page_end")
    op.drop_column("standards", "source_page_start")
    op.drop_column("standards", "source_revision")
    op.drop_column("standards", "source_record_id")
    op.drop_column("standards", "source_dataset")
