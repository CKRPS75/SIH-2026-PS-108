"""Add full parsed-corpus standard fields."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260916_01"
down_revision = "20260912_01"
branch_labels = None
depends_on = None

json_storage_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("standards", sa.Column("standard_code_norm", sa.String(255)))
    op.add_column("standards", sa.Column("page_start", sa.Integer()))
    op.add_column("standards", sa.Column("page_end", sa.Integer()))
    op.add_column("standards", sa.Column("scope", sa.Text()))
    op.add_column("standards", sa.Column("standard_kind", sa.String(64)))
    op.add_column("standards", sa.Column("canonical_product", sa.String(128)))
    op.add_column(
        "standards",
        sa.Column(
            "product_aliases",
            json_storage_type,
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    op.add_column("standards", sa.Column("family", sa.String(128)))
    op.add_column(
        "standards",
        sa.Column(
            "source_provenance",
            json_storage_type,
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    op.create_index("ix_standards_standard_code_norm", "standards", ["standard_code_norm"])
    op.create_index("ix_standards_standard_kind", "standards", ["standard_kind"])
    op.create_index("ix_standards_canonical_product", "standards", ["canonical_product"])
    op.create_index("ix_standards_family", "standards", ["family"])


def downgrade() -> None:
    op.drop_index("ix_standards_family", table_name="standards")
    op.drop_index("ix_standards_canonical_product", table_name="standards")
    op.drop_index("ix_standards_standard_kind", table_name="standards")
    op.drop_index("ix_standards_standard_code_norm", table_name="standards")
    op.drop_column("standards", "source_provenance")
    op.drop_column("standards", "family")
    op.drop_column("standards", "product_aliases")
    op.drop_column("standards", "canonical_product")
    op.drop_column("standards", "standard_kind")
    op.drop_column("standards", "scope")
    op.drop_column("standards", "page_end")
    op.drop_column("standards", "page_start")
    op.drop_column("standards", "standard_code_norm")
