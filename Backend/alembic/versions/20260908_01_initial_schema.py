"""Create StandardWise Phase 1 data tables."""

import sqlalchemy as sa

from alembic import op

revision = "20260908_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    uuid_type = sa.Uuid()
    op.create_table(
        "standards",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("standard_id", sa.String(255), nullable=False),
        sa.Column("canonical_id", sa.String(255), nullable=False),
        sa.Column("base_code", sa.String(64), nullable=False),
        sa.Column("part", sa.String(64)),
        sa.Column("section", sa.String(64)),
        sa.Column("title", sa.String(1000), nullable=False),
        sa.Column("scope_text", sa.Text()),
        sa.Column("division", sa.String(128)),
        sa.Column("publication_year", sa.Integer()),
        sa.Column("status", sa.String(32), nullable=False, server_default="ACTIVE"),
        sa.Column("supersedes_id", uuid_type, sa.ForeignKey("standards.id")),
        sa.Column("replaced_by_id", uuid_type, sa.ForeignKey("standards.id")),
        sa.Column("amendments_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("keywords", sa.JSON(), nullable=False),
        sa.Column("source_url", sa.Text()),
        sa.Column("last_verified_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_standards_standard_id", "standards", ["standard_id"])
    op.create_index("ix_standards_canonical_id", "standards", ["canonical_id"])
    op.create_index("ix_standards_base_code", "standards", ["base_code"])
    op.create_index("ix_standards_status", "standards", ["status"])
    op.create_table(
        "standard_amendments",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("standard_id", uuid_type, sa.ForeignKey("standards.id"), nullable=False),
        sa.Column("amendment_no", sa.String(64), nullable=False),
        sa.Column("publication_date", sa.Date()),
        sa.Column("summary", sa.Text()),
        sa.Column("source_url", sa.Text()),
        sa.UniqueConstraint("standard_id", "amendment_no", name="uq_standard_amendment"),
    )
    op.create_index("ix_standard_amendments_standard_id", "standard_amendments", ["standard_id"])
    op.create_table(
        "qco_rules",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("standard_id", uuid_type, sa.ForeignKey("standards.id")),
        sa.Column("product_category", sa.String(256)),
        sa.Column("hs_code", sa.String(32)),
        sa.Column("scheme", sa.String(64), nullable=False),
        sa.Column("notifying_ministry", sa.String(256), nullable=False),
        sa.Column("order_ref", sa.String(256), nullable=False),
        sa.Column("effective_from", sa.Date()),
        sa.Column("effective_to", sa.Date()),
        sa.Column("status", sa.String(32), nullable=False, server_default="ACTIVE"),
        sa.Column("source_url", sa.Text()),
        sa.UniqueConstraint("order_ref"),
    )
    op.create_index("ix_qco_rules_standard_id", "qco_rules", ["standard_id"])
    op.create_index("ix_qco_rules_product_category", "qco_rules", ["product_category"])
    op.create_index("ix_qco_rules_hs_code", "qco_rules", ["hs_code"])
    op.create_table(
        "tenders",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("file_type", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="UPLOADED"),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("language", sa.String(32)),
        sa.Column("source_hash", sa.String(128), nullable=False),
        sa.UniqueConstraint("source_hash"),
    )
    op.create_table(
        "tender_items",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("tender_id", uuid_type, sa.ForeignKey("tenders.id"), nullable=False),
        sa.Column("item_no", sa.String(64)),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("page_or_row_ref", sa.String(128), nullable=False),
        sa.Column("normalized_text", sa.Text()),
        sa.Column("extracted_specs", sa.JSON(), nullable=False),
    )
    op.create_index("ix_tender_items_tender_id", "tender_items", ["tender_id"])
    op.create_table(
        "analysis_results",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("tender_item_id", uuid_type, sa.ForeignKey("tender_items.id")),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("normalized_query", sa.Text()),
        sa.Column("primary_results", sa.JSON(), nullable=False),
        sa.Column("allied_results", sa.JSON(), nullable=False),
        sa.Column("compliance", sa.JSON(), nullable=False),
        sa.Column("trace", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_analysis_results_tender_item_id", "analysis_results", ["tender_item_id"])
    op.create_table(
        "feedback",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "analysis_result_id",
            uuid_type,
            sa.ForeignKey("analysis_results.id"),
            nullable=False,
        ),
        sa.Column("selected_standard_id", uuid_type, sa.ForeignKey("standards.id")),
        sa.Column("rating", sa.Integer()),
        sa.Column("comment", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_feedback_analysis_result_id", "feedback", ["analysis_result_id"])


def downgrade() -> None:
    op.drop_index("ix_feedback_analysis_result_id", table_name="feedback")
    op.drop_table("feedback")
    op.drop_index("ix_analysis_results_tender_item_id", table_name="analysis_results")
    op.drop_table("analysis_results")
    op.drop_index("ix_tender_items_tender_id", table_name="tender_items")
    op.drop_table("tender_items")
    op.drop_table("tenders")
    op.drop_index("ix_qco_rules_hs_code", table_name="qco_rules")
    op.drop_index("ix_qco_rules_product_category", table_name="qco_rules")
    op.drop_index("ix_qco_rules_standard_id", table_name="qco_rules")
    op.drop_table("qco_rules")
    op.drop_index("ix_standard_amendments_standard_id", table_name="standard_amendments")
    op.drop_table("standard_amendments")
    op.drop_index("ix_standards_status", table_name="standards")
    op.drop_index("ix_standards_base_code", table_name="standards")
    op.drop_index("ix_standards_canonical_id", table_name="standards")
    op.drop_index("ix_standards_standard_id", table_name="standards")
    op.drop_table("standards")
