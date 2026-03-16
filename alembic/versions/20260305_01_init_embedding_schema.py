"""init embedding management schema

Revision ID: 20260305_01
Revises: 
Create Date: 2026-03-05 13:30:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "20260305_01"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


problem_asset_file_type = postgresql.ENUM(
    "COMMON_JSON",
    "PROBLEM_MD",
    "QUIZ_JSON",
    "EMBED_JSON",
    name="problem_asset_file_type",
    create_type=False,
)

paragraph_type = postgresql.ENUM(
    "BACKGROUND",
    "GOAL",
    "STRATEGY",
    "INSIGHT",
    name="paragraph_type",
    create_type=False,
)

review_status = postgresql.ENUM(
    "DRAFT",
    "IN_REVIEW",
    "APPROVED",
    name="review_status",
    create_type=False,
)

embedding_status = postgresql.ENUM(
    "PENDING",
    "PROCESSING",
    "DONE",
    "FAILED",
    name="embedding_status",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    problem_asset_file_type.create(bind, checkfirst=True)
    paragraph_type.create(bind, checkfirst=True)
    review_status.create(bind, checkfirst=True)
    embedding_status.create(bind, checkfirst=True)

    op.create_table(
        "problem",
        sa.Column("problem_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("difficulty", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("problem_id"),
    )

    op.create_table(
        "problem_asset",
        sa.Column("problem_id", sa.Integer(), nullable=False),
        sa.Column("file_type", problem_asset_file_type, nullable=False),
        sa.Column("json_body", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("md_body", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["problem_id"], ["problem.problem_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("problem_id", "file_type"),
    )

    op.create_table(
        "embedding_job",
        sa.Column("problem_id", sa.Integer(), nullable=False),
        sa.Column("paragraph_type", paragraph_type, nullable=False),
        sa.Column("review_status", review_status, server_default=sa.text("'DRAFT'"), nullable=False),
        sa.Column("embedding_status", embedding_status, server_default=sa.text("'PENDING'"), nullable=False),
        sa.Column("source_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["problem_id"], ["problem.problem_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("problem_id", "paragraph_type"),
    )


def downgrade() -> None:
    op.drop_table("embedding_job")
    op.drop_table("problem_asset")
    op.drop_table("problem")

    bind = op.get_bind()
    embedding_status.drop(bind, checkfirst=True)
    review_status.drop(bind, checkfirst=True)
    paragraph_type.drop(bind, checkfirst=True)
    problem_asset_file_type.drop(bind, checkfirst=True)
