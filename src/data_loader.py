"""
Data Loader module for Sports Team Partitioning.
Supports automatic column detection for both Thai and English datasets.
"""
import pandas as pd
from typing import Optional, Dict

class DataLoader:
    """Loads and standardizes participant data from CSV files."""
    
    THAI_COLUMN_MAPPINGS = {
        'id': ['id', 'รหัส', 'รหัสประจำตัว', 'เลขที่', 'ID', 'emp_id'],
        'name': ['name', 'ชื่อ-นามสกุล', 'ชื่อ', 'Name', 'full_name'],
        'department': ['department', 'สาขาวิชา', 'สาขา', 'อาจารย์ที่ปรึกษา', 'แผนก', 'สังกัด', 'ฝ่าย'],
        'sports_skill': ['sports_skill', 'คะแนน', 'skill', 'score', 'sports', 'ทักษะกีฬา']
    }

    @classmethod
    def load(cls, file_path: str, custom_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Loads CSV and returns standardized DataFrame with columns:
        ['id', 'name', 'department', 'sports_skill']
        """
        # Try loading with utf-8-sig first, fallback to cp874/tis-620
        df = None
        for enc in ['utf-8-sig', 'utf-8', 'cp874', 'tis-620']:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                break
            except Exception:
                continue
                
        if df is None:
            raise ValueError(f"Could not read {file_path} with supported encodings (utf-8, cp874, tis-620).")
            
        # Strip whitespace from column names
        df.columns = [str(c).strip() for c in df.columns]

        resolved_cols = {}
        if custom_mapping:
            for std_col, actual_col in custom_mapping.items():
                if actual_col in df.columns:
                    resolved_cols[std_col] = actual_col
                    
        # Auto-detect missing columns
        for std_col, candidates in cls.THAI_COLUMN_MAPPINGS.items():
            if std_col not in resolved_cols:
                for c in candidates:
                    if c in df.columns:
                        resolved_cols[std_col] = c
                        break

        # Check required columns
        for req in ['id', 'name', 'department', 'sports_skill']:
            if req not in resolved_cols:
                raise KeyError(
                    f"Required field '{req}' could not be matched. Available columns: {list(df.columns)}. "
                    f"Please provide custom_mapping."
                )

        standardized_df = pd.DataFrame({
            'id': df[resolved_cols['id']].astype(str),
            'name': df[resolved_cols['name']].astype(str),
            'department': df[resolved_cols['department']].astype(str),
            'sports_skill': pd.to_numeric(df[resolved_cols['sports_skill']], errors='coerce').fillna(50.0)
        })

        return standardized_df
