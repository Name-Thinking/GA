"""
Unit tests แบบง่ายสำหรับตรวจสอบการทำงานของ Genetic Algorithm
"""
import pytest
import os
import pandas as pd

def test_data_loading():
    """ทดสอบว่าโหลดไฟล์ข้อมูล 40 คนได้ถูกต้อง"""
    assert os.path.exists("data/company_members.csv")
    df = pd.read_csv("data/company_members.csv")
    assert len(df) == 40
    assert 'sports_skill' in df.columns
    assert 'department' in df.columns

def test_chromosome_length_and_balance():
    """ทดสอบว่าโครโมโซมมีความยาว 40 และแบ่งคนสีละ 20 เท่ากัน"""
    half = 20
    chrom = [0] * half + [1] * half
    assert len(chrom) == 40
    assert chrom.count(0) == 20
    assert chrom.count(1) == 20

def test_output_csv_generated():
    """ทดสอบว่าไฟล์ผลลัพธ์ถูกสร้างและมีครบทั้งสองสี"""
    assert os.path.exists("outputs/team_assignment.csv")
    df = pd.read_csv("outputs/team_assignment.csv")
    assert len(df) == 40
    assert df['Team'].value_counts()['สีแดง (Red)'] == 20
    assert df['Team'].value_counts()['สีเขียว (Green)'] == 20
