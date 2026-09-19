"""Add parsed-corpus metadata enrichment fields."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260916_02"
down_revision = "20260916_01"
branch_labels = None
depends_on = None

json_storage_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("standards", sa.Column("product_subtype", sa.String(128)))
    op.add_column("standards", sa.Column("primary_subject", sa.String(512)))
    op.add_column("standards", sa.Column("material", sa.String(255)))
    op.add_column("standards", sa.Column("application", sa.String(255)))
    op.add_column("standards", sa.Column("function", sa.String(255)))
    op.add_column(
        "standards",
        sa.Column(
            "applies_to_product_families",
            json_storage_type,
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    op.add_column("standards", sa.Column("product_confidence", sa.String(32)))
    op.add_column("standards", sa.Column("metadata_confidence", sa.String(32)))
    op.add_column(
        "standards",
        sa.Column(
            "metadata_evidence",
            json_storage_type,
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.add_column("standards", sa.Column("metadata_derivation_method", sa.String(128)))
    op.add_column(
        "standards",
        sa.Column(
            "review_required",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column("standards", sa.Column("review_reason", sa.Text()))
    op.add_column("standards", sa.Column("raw_title", sa.Text()))
    op.add_column("standards", sa.Column("canonical_title_expanded", sa.Text()))
    op.add_column(
        "standards",
        sa.Column(
            "title_evidence",
            json_storage_type,
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    op.add_column("standards", sa.Column("scope_reconstructed", sa.Text()))
    op.add_column("standards", sa.Column("scope_source", sa.String(128)))
    op.create_index("ix_standards_product_subtype", "standards", ["product_subtype"])
    op.create_index("ix_standards_primary_subject", "standards", ["primary_subject"])
    op.create_index("ix_standards_material", "standards", ["material"])
    op.create_index("ix_standards_application", "standards", ["application"])
    op.create_index("ix_standards_function", "standards", ["function"])
    op.create_index("ix_standards_product_confidence", "standards", ["product_confidence"])
    op.create_index("ix_standards_metadata_confidence", "standards", ["metadata_confidence"])
    op.create_index("ix_standards_review_required", "standards", ["review_required"])


def downgrade() -> None:
    op.drop_index("ix_standards_review_required", table_name="standards")
    op.drop_index("ix_standards_metadata_confidence", table_name="standards")
    op.drop_index("ix_standards_product_confidence", table_name="standards")
    op.drop_index("ix_standards_function", table_name="standards")
    op.drop_index("ix_standards_application", table_name="standards")
    op.drop_index("ix_standards_material", table_name="standards")
    op.drop_index("ix_standards_primary_subject", table_name="standards")
    op.drop_index("ix_standards_product_subtype", table_name="standards")
    op.drop_column("standards", "scope_source")
    op.drop_column("standards", "scope_reconstructed")
    op.drop_column("standards", "title_evidence")
    op.drop_column("standards", "canonical_title_expanded")
    op.drop_column("standards", "raw_title")
    op.drop_column("standards", "review_reason")
    op.drop_column("standards", "review_required")
    op.drop_column("standards", "metadata_derivation_method")
    op.drop_column("standards", "metadata_evidence")
    op.drop_column("standards", "metadata_confidence")
    op.drop_column("standards", "product_confidence")
    op.drop_column("standards", "applies_to_product_families")
    op.drop_column("standards", "function")
    op.drop_column("standards", "application")
    op.drop_column("standards", "material")
    op.drop_column("standards", "primary_subject")
    op.drop_column("standards", "product_subtype")
