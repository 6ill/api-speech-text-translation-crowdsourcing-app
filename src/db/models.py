from sqlmodel import SQLModel, Field, Column, Relationship
import sqlalchemy.dialects.postgresql as pg
from uuid import UUID, uuid4
from datetime import datetime, date, timezone
from enum import StrEnum
from typing import List, Optional


class Role(StrEnum):
    USER = "user"
    ADMIN = "admin"


class FileStatus(StrEnum):
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    TRANSLATING = "translating"
    TRANSLATED = "translated"


class CorrectionStatus(StrEnum):
    PENDING = "pending"
    REJECTED = "rejected"
    APPROVED = "approved"

class PipelineTaskType(StrEnum):
    ASR = "asr"
    MT = "mt"

class PipelineRunStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped" 

class PipelineConfig(SQLModel, table=True):
    __tablename__ = "pipeline_configs"

    id: UUID = Field(sa_column=Column(pg.UUID, primary_key=True, default=uuid4))
    
    # Only one unique config per type
    task_type: PipelineTaskType = Field(
        sa_column=Column(pg.ENUM(PipelineTaskType), unique=True, index=True)
    )
    
    is_active: bool = Field(default=True)
    
    # example: "0 3 * * 0" every week at 3AM)
    cron_schedule: str = Field(max_length=50, default="0 3 * * 0")
    
    min_samples_required: int = Field(default=100)
    
    learning_rate: float = Field(default=2e-5)
    num_epochs: int = Field(default=3)
    batch_size: int = Field(default=4)
    
    evaluation_dataset_storage_key: str = Field(max_length=255)
    
    created_at: datetime = Field(
        sa_column=Column(
            pg.TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)
        )
    )
    updated_at: datetime = Field(
        sa_column=Column(
            pg.TIMESTAMP(timezone=True), 
            default=lambda: datetime.now(timezone.utc),
            onupdate=lambda: datetime.now(timezone.utc)
        )
    )

class PipelineRunLog(SQLModel, table=True):
    __tablename__ = "pipeline_run_logs"

    id: UUID = Field(sa_column=Column(pg.UUID, primary_key=True, default=uuid4))
    
    config_id: UUID = Field(foreign_key="pipeline_configs.id")
    
    mlflow_run_id: Optional[str] = Field(default=None, max_length=100)
    
    status: PipelineRunStatus = Field(sa_column=Column(pg.ENUM(PipelineRunStatus)))
    
    data_samples_used: int = Field(default=0)
    
    # example: {"wer": 0.25, "bleu": 24.5}
    metrics_baseline: Optional[dict] = Field(sa_column=Column(pg.JSON, nullable=True))
    metrics_new_model: Optional[dict] = Field(sa_column=Column(pg.JSON, nullable=True))
    
    message: Optional[str] = Field(sa_column=Column(pg.TEXT, nullable=True))
    
    start_time: datetime = Field(
        sa_column=Column(
            pg.TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)
        )
    )
    end_time: Optional[datetime] = Field(
        sa_column=Column(pg.TIMESTAMP(timezone=True), nullable=True)
    )

class User(SQLModel, table=True):
    __tablename__ = "users"

    id: UUID = Field(sa_column=Column(pg.UUID, primary_key=True, default=uuid4))
    email: str = Field(unique=True, index=True, max_length=255)
    password_hash: str = Field(exclude=True, max_length=255)
    role: Role = Field(
        sa_column=Column(
            pg.ENUM(Role),
            default=Role.USER,
            index=True,
            server_default="USER",
        )
    )
    created_at: datetime = Field(
        sa_column=Column(
            pg.TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)
        )
    )

    files: List["File"] = Relationship(
        back_populates="user", sa_relationship_kwargs={"lazy": "selectin", "cascade": "all, delete-orphan"}
    )

    def __repr__(self) -> str:
        return f"<User {self.email}>"


class File(SQLModel, table=True):
    __tablename__ = "files"

    id: UUID = Field(sa_column=Column(pg.UUID, primary_key=True, default=uuid4))

    user_id: UUID = Field(foreign_key="users.id", index=True, ondelete="CASCADE")
    status: FileStatus = Field(sa_column=Column(pg.ENUM(FileStatus), index=True))
    file_name: str = Field(max_length=255)
    storage_bucket: str = Field(max_length=255)
    storage_key: str = Field(max_length=255)
    mime_type: str = Field(max_length=100)
    file_size: int = Field(sa_column=Column(pg.BIGINT))
    duration_seconds: float
    created_at: datetime = Field(
        sa_column=Column(
            pg.TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)
        )
    )
    speaker_id: Optional[UUID] = Field(foreign_key="people.id", index=True, default=None)

    user: User = Relationship(back_populates="files")
    segments: List["Segment"] = Relationship(
        back_populates="file", sa_relationship_kwargs={"lazy": "selectin", "cascade": "all, delete-orphan"}
    )
    speaker: Optional["People"] = Relationship(back_populates="files")

    def __repr__(self) -> str:
        return f"<File {self.file_name}>"


class Segment(SQLModel, table=True):
    __tablename__ = "segments"

    id: UUID = Field(sa_column=Column(pg.UUID, primary_key=True, default=uuid4))

    start_timestamp: float
    end_timestamp: float
    transcription_text: str = Field(sa_column=Column(pg.TEXT))
    translation_text: Optional[str] = Field(sa_column=Column(pg.TEXT, nullable=True))
    file_id: UUID = Field(foreign_key="files.id", index=True, ondelete="CASCADE")

    file: File = Relationship(back_populates="segments")
    transcription_corrections: List["TranscriptionCorrection"] = Relationship(
        back_populates="segment", sa_relationship_kwargs={"lazy": "selectin", "cascade": "all, delete-orphan"}
    )
    translation_corrections: List["TranslationCorrection"] = Relationship(
        back_populates="segment", sa_relationship_kwargs={"lazy": "selectin", "cascade": "all, delete-orphan"}
    )

    def __repr__(self):
        return (
            f"<Segment {self.start_timestamp}-{self.end_timestamp} in {self.file_id}>"
        )


class TranscriptionCorrection(SQLModel, table=True):
    __tablename__ = "transcription_corrections"

    id: UUID = Field(sa_column=Column(pg.UUID, primary_key=True, default=uuid4))

    # Foreign Key ke Segment (ondelete=CASCADE)
    segment_id: UUID = Field(foreign_key="segments.id", index=True, ondelete="CASCADE")
    original_text: str = Field(sa_column=Column(pg.TEXT))
    corrected_text: str = Field(sa_column=Column(pg.TEXT))
    status: CorrectionStatus = Field(
        sa_column=Column(
            pg.ENUM(CorrectionStatus),
            default=CorrectionStatus.PENDING,
            index=True,
            server_default="PENDING",
        )
    )
    used_for_training: bool = Field(default=False)

    segment: Segment = Relationship(back_populates="transcription_corrections")


class TranslationCorrection(SQLModel, table=True):
    __tablename__ = "translation_corrections"

    id: UUID = Field(sa_column=Column(pg.UUID, primary_key=True, default=uuid4))

    segment_id: UUID = Field(foreign_key="segments.id", index=True, ondelete="CASCADE")
    original_text: str = Field(sa_column=Column(pg.TEXT))
    corrected_text: str = Field(sa_column=Column(pg.TEXT))
    status: CorrectionStatus = Field(
        sa_column=Column(
            pg.ENUM(CorrectionStatus),
            default=CorrectionStatus.PENDING,
            index=True,
            server_default="PENDING",
        )
    )
    used_for_training: bool = Field(default=False)

    segment: Segment = Relationship(back_populates="translation_corrections")

class People(SQLModel, table=True):
    __tablename__ = "people"
    id: UUID = Field(sa_column=Column(pg.UUID, primary_key=True, default=uuid4))
    
    email: str = Field(unique=True, index=True, max_length=255)
    name: str = Field(index=True, max_length=255)
    
    files: List[File] = Relationship(
        back_populates="speaker", sa_relationship_kwargs={"lazy": "selectin"}
    )


AllModels = [User, File, Segment, TranscriptionCorrection, TranslationCorrection]