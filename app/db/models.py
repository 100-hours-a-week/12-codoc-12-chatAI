import enum

from sqlalchemy import CHAR, ForeignKey, Integer, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.sqltypes import DateTime

from app.db.base import Base


class ProblemAssetFileType(str, enum.Enum):
    COMMON_JSON = "COMMON_JSON"
    PROBLEM_MD = "PROBLEM_MD"
    QUIZ_JSON = "QUIZ_JSON"
    EMBED_JSON = "EMBED_JSON"


class ParagraphType(str, enum.Enum):
    BACKGROUND = "BACKGROUND"
    GOAL = "GOAL"
    STRATEGY = "STRATEGY"
    INSIGHT = "INSIGHT"


class ReviewStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    IN_REVIEW = "IN_REVIEW"
    APPROVED = "APPROVED"


class EmbeddingStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


class Problem(Base):
    __tablename__ = "problem"

    problem_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class ProblemAsset(Base):
    __tablename__ = "problem_asset"

    problem_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("problem.problem_id", ondelete="CASCADE"),
        primary_key=True,
    )
    file_type: Mapped[ProblemAssetFileType] = mapped_column(
        SQLEnum(
            ProblemAssetFileType,
            name="problem_asset_file_type",
            native_enum=True,
            create_constraint=True,
        ),
        primary_key=True,
    )
    json_body: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    md_body: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class EmbeddingJob(Base):
    __tablename__ = "embedding_job"

    problem_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("problem.problem_id", ondelete="CASCADE"),
        primary_key=True,
    )
    paragraph_type: Mapped[ParagraphType] = mapped_column(
        SQLEnum(
            ParagraphType,
            name="paragraph_type",
            native_enum=True,
            create_constraint=True,
        ),
        primary_key=True,
    )
    review_status: Mapped[ReviewStatus] = mapped_column(
        SQLEnum(
            ReviewStatus,
            name="review_status",
            native_enum=True,
            create_constraint=True,
        ),
        nullable=False,
        server_default=ReviewStatus.DRAFT.value,
    )
    embedding_status: Mapped[EmbeddingStatus] = mapped_column(
        SQLEnum(
            EmbeddingStatus,
            name="embedding_status",
            native_enum=True,
            create_constraint=True,
        ),
        nullable=False,
        server_default=EmbeddingStatus.PENDING.value,
    )
    source_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
