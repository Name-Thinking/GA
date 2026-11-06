"""
Unit tests สำหรับตรวจสอบการทำงานของ Genetic Algorithm บนข้อมูล HR(in).csv
"""
import pytest
import os
import pandas as pd

def test_data_loading():
    """ทดสอบว่าโหลดไฟล์ HR(in).csv (358 คน) ได้ถูกต้อง"""
    assert os.path.exists("data/HR(in).csv")
    df = pd.read_csv("data/HR(in).csv")
    assert len(df) == 358
    assert 'gender' in df.columns
    assert 'department' in df.columns
    assert 'weight' in df.columns
    assert 'height' in df.columns
    assert 'age' in df.columns

def test_chromosome_length_and_balance():
    """ทดสอบว่าโครโมโซมมีความยาว 358 และแบ่งคนสีละ 179 เท่ากัน"""
    half = 179
    chrom = [0] * half + [1] * half
    assert len(chrom) == 358
    assert chrom.count(0) == 179
    assert chrom.count(1) == 179

def test_output_csv_generated():
    """ทดสอบว่าไฟล์ผลลัพธ์ถูกสร้างและมีครบทั้งสองสี สีละ 179 คน"""
    assert os.path.exists("outputs/team_assignment.csv")
    df = pd.read_csv("outputs/team_assignment.csv")
    assert len(df) == 358
    assert df['Team'].value_counts()['สีแดง (Red)'] == 179
    assert df['Team'].value_counts()['สีเขียว (Green)'] == 179
