"""seed initial pipeline configs

Revision ID: cdfb8ad0e526
Revises: f9d4387e4161
Create Date: 2026-03-30 11:56:16.720839

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cdfb8ad0e526'
down_revision: Union[str, Sequence[str], None] = 'f9d4387e4161'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO pipeline_configs (
            id,
            task_type,
            is_active,
            cron_schedule,
            min_samples_required,
            num_epochs,
            batch_size,
            learning_rate,
            evaluation_dataset_storage_key,
            created_at,
            updated_at
        ) VALUES 
        (
            gen_random_uuid(), 
            'ASR',
            true,
            '0 2 * * 0',
            500,
            3,
            8,
            0.00001,
            'asr/asr_test_set.zip',
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        ),
        (
            gen_random_uuid(), 
            'MT',
            true,
            '0 4 * * 0',
            1000,
            3,
            4,
            0.0002,
            'mt/mt_test_set.jsonl',
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        );
    """)


def downgrade() -> None:
    """Remove the initial pipeline configuration data."""
    
    op.execute("""
        DELETE FROM pipeline_configs WHERE task_type IN ('asr', 'mt');
    """)
