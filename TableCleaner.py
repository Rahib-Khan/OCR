"""
Pattern-Based Intelligent Table Cleaner
Uses actual patterns from the historical document structure
FIXED: Auto-detects column offset and handles empty leading columns
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict


class IntelligentTableCleaner:
    """
    Advanced cleaner that recognizes patterns specific to 1950s air pollution tables.
    """
    
    def __init__(self):
        # US state abbreviations that appear in station names
        self.state_abbrevs = {
            'CONN', 'MASS', 'NH', 'NJ', 'NY', 'RI', 'VT',
            'PA', 'MD', 'VA', 'WVA', 'NC', 'SC', 'GA', 'FLA',
            'TENN', 'KY', 'OHIO', 'MICH', 'IND', 'WIS', 'ILL',
            'IOWA', 'MO', 'KANSAS', 'NEB', 'ND', 'SD',
            'TEXAS', 'OKLA', 'ARK', 'LA',
            'COL', 'ARIZ', 'NEVADA', 'CALIF', 'WASH', 'IDAHO'
        }
        
        self.column_offset = 0  # Will be auto-detected
        
    def detect_column_offset(self, df) -> int:
        """
        Auto-detect which column contains station names.
        Some tables have empty Column_0, so stations start at Column_1.
        """
        print("Auto-detecting column structure...")
        
        # Check first few non-empty columns
        for col_idx in range(min(3, len(df.columns))):
            col = df.iloc[:, col_idx]
            
            # Count non-empty text values (longer than 3 chars)
            text_values = col.dropna().astype(str)
            substantial_text = sum(1 for v in text_values if len(v.strip()) > 3)
            
            if substantial_text > 10:  # If we find 10+ substantial text values
                print(f"  → Station names detected in column index {col_idx}")
                return col_idx
        
        print(f"  → Using default column index 0")
        return 0
    
    def is_numeric_data(self, value) -> bool:
        """Check if value looks like numeric data."""
        if pd.isna(value):
            return False
        text = str(value).strip()
        if not text:
            return False
        # Check if mostly digits (allowing for OCR errors like 'i', 'l', 'O')
        digit_like = sum(c.isdigit() or c in 'ilIoOB' for c in text)
        return len(text) > 0 and digit_like / len(text) > 0.5
    
    def clean_numeric(self, value) -> str:
        """Clean numeric value from OCR artifacts."""
        if pd.isna(value) or str(value).strip() == '':
            return ''
        
        text = str(value).strip()
        
        # Common OCR substitutions
        replacements = {
            'i': '1', 'l': '1', 'I': '1',
            'O': '0', 'o': '0',
            'B': '8',
            'S': '5', 's': '5',
            'Z': '2', 'z': '2',
            'g': '9',
            'Les': '105',  # Common OCR error
            'Bs': '85',
            'i2i': '121',
            'ie': '16',
            'ey': '67',
            'i1': '11',
            '1i': '11',
            'i1i': '111'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove non-numeric characters except decimal point
        text = re.sub(r'[^0-9.]', '', text)
        
        # Handle multiple decimal points (keep first)
        if text.count('.') > 1:
            parts = text.split('.')
            text = parts[0] + '.' + ''.join(parts[1:])
        
        return text
    
    def normalize_site_code(self, site_text: str) -> str:
        """Extract and normalize site code."""
        if not site_text or site_text == 'nan':
            return ''
        
        # Remove leading numbers that are row indices (like "1 54" -> "54")
        parts = str(site_text).strip().split()
        if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) <= 2:
            return parts[1]
        
        # Extract last meaningful number
        numbers = re.findall(r'\d+', str(site_text))
        return numbers[-1] if numbers else ''
    
    def normalize_years(self, years_text: str) -> str:
        """Normalize year representation."""
        if not years_text or years_text == 'nan':
            return ''
        
        text = str(years_text).strip()
        
        # If multiple years separated by space, use hyphen
        # E.g., "54 56" -> "54-56"
        parts = text.split()
        if len(parts) == 2 and all(p.isdigit() and len(p) <= 2 for p in parts):
            return f"{parts[0]}-{parts[1]}"
        
        # Extract all year-like numbers (2 digits, 50-59)
        years = re.findall(r'5[0-9]', text)
        if len(years) > 1:
            return '-'.join(years)
        elif years:
            return years[0]
        
        return text
    
    def is_state_continuation(self, text: str) -> bool:
        """Check if text is just a state abbreviation (continuation)."""
        if not text or text == 'nan':
            return False
        text_upper = text.upper().strip()
        
        # Check against known state abbreviations
        return text_upper in self.state_abbrevs
    
    def merge_station_name(self, parts: list) -> str:
        """Intelligently merge station name parts."""
        if not parts:
            return ''
        
        # Remove empty parts
        parts = [p.strip() for p in parts if p and p.strip() and p != 'nan']
        
        # Join with single space
        merged = ' '.join(parts)
        
        # Fix spacing around state codes
        merged = re.sub(r'\s+N\s+J\s*', ' NJ ', merged)
        merged = re.sub(r'\s+N\s+Y\s*', ' NY ', merged)
        merged = re.sub(r'\s+N\s+H\s*', ' NH ', merged)
        merged = re.sub(r'\s+R\s+I\s*', ' RI ', merged)
        
        return merged.strip()
    
    def has_valid_data_row(self, row, data_start_idx) -> bool:
        """
        Check if row has valid data values.
        A valid row should have at least 2 numeric values.
        """
        if data_start_idx >= len(row):
            return False
            
        data_cells = row.iloc[data_start_idx:]
        numeric_count = sum(1 for cell in data_cells if self.is_numeric_data(cell))
        return numeric_count >= 2
    
    def clean_table(self, input_csv: str, output_csv: str = None):
        """
        Main cleaning pipeline.
        
        Args:
            input_csv: Path to input CSV
            output_csv: Path to save cleaned CSV (optional)
        """
        print(f"\n{'='*80}")
        print(f"INTELLIGENT PATTERN-BASED TABLE CLEANER")
        print(f"{'='*80}\n")
        
        # Load data
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} rows from: {input_csv}")
        print(f"Columns: {list(df.columns)}\n")
        
        # Auto-detect column offset
        self.column_offset = self.detect_column_offset(df)
        
        # Calculate where data columns start (station + site + years + data)
        data_start_idx = self.column_offset + 3
        
        print(f"Column mapping:")
        print(f"  Station names: Column index {self.column_offset}")
        print(f"  Site codes:    Column index {self.column_offset + 1}")
        print(f"  Years:         Column index {self.column_offset + 2}")
        print(f"  Data starts:   Column index {data_start_idx}\n")
        
        # Statistics
        stats = {
            'input_rows': len(df),
            'header_removed': 0,
            'empty_removed': 0,
            'stations_merged': 0,
            'output_rows': 0
        }
        
        # Clean data
        cleaned_data = []
        current_station = ''
        current_site = ''
        current_years = ''
        station_buffer = []  # Buffer for multi-row station names
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Get raw column values using detected offset
            col_station = str(row.iloc[self.column_offset]).strip() if pd.notna(row.iloc[self.column_offset]) else ''
            col_site = str(row.iloc[self.column_offset + 1]).strip() if self.column_offset + 1 < len(row) and pd.notna(row.iloc[self.column_offset + 1]) else ''
            col_years = str(row.iloc[self.column_offset + 2]).strip() if self.column_offset + 2 < len(row) and pd.notna(row.iloc[self.column_offset + 2]) else ''
            
            # Clean up 'nan' strings
            if col_station == 'nan': col_station = ''
            if col_site == 'nan': col_site = ''
            if col_years == 'nan': col_years = ''
            
            # Skip completely empty rows
            if not any(str(v).strip() and str(v) != 'nan' for v in row.values if pd.notna(v)):
                stats['empty_removed'] += 1
                continue
            
            # Skip if header row (contains keywords)
            row_text = ' '.join(str(v) for v in row.values if pd.notna(v) and str(v) != 'nan').upper()
            if any(kw in row_text for kw in ['STATION LOCATION', 'MICROGRAMS', 'FREQUENCY', 'SAMPLES', 'COLUMN_', 'TABLE']):
                stats['header_removed'] += 1
                continue
            
            # Check if this row has actual data
            has_data = self.has_valid_data_row(row, data_start_idx)
            
            # Case 1: New station with data
            if col_station and len(col_station) > 2 and has_data:
                # This is a new station
                station_buffer = [col_station]
                current_station = col_station
                
                # Update site and years if present
                if col_site:
                    current_site = self.normalize_site_code(col_site)
                if col_years:
                    current_years = self.normalize_years(col_years)
                
                # Extract data row
                data_row = self.extract_data_row(
                    row, current_station, current_site, current_years, data_start_idx
                )
                cleaned_data.append(data_row)
                stats['output_rows'] += 1
                
            # Case 2: Station name continuation (state abbrev only, no data)
            elif col_station and self.is_state_continuation(col_station) and not has_data:
                station_buffer.append(col_station)
                current_station = self.merge_station_name(station_buffer)
                stats['stations_merged'] += 1
                
            # Case 3: Additional data row for same station
            elif has_data and current_station:
                # Update site/years if present
                if col_site:
                    current_site = self.normalize_site_code(col_site)
                if col_years:
                    current_years = self.normalize_years(col_years)
                
                # Extract data row
                data_row = self.extract_data_row(
                    row, current_station, current_site, current_years, data_start_idx
                )
                cleaned_data.append(data_row)
                stats['output_rows'] += 1
                
            # Case 4: Empty/invalid row
            else:
                stats['empty_removed'] += 1
        
        # Create output DataFrame
        if not cleaned_data:
            print("\n⚠ WARNING: No data rows were extracted!")
            print("\nShowing first 10 rows of input for debugging:")
            print(df.head(10).to_string())
            print("\nShowing sample of non-empty values:")
            for i in range(min(5, len(df))):
                non_empty = [str(v) for v in df.iloc[i].values if pd.notna(v) and str(v).strip() and str(v) != 'nan']
                if non_empty:
                    print(f"  Row {i}: {non_empty[:10]}")
            return pd.DataFrame()
        
        output_df = pd.DataFrame(cleaned_data)
        
        # Print statistics
        print(f"\n{'='*80}")
        print(f"Cleaning Statistics:")
        print(f"  Input rows:           {stats['input_rows']}")
        print(f"  Header rows removed:  {stats['header_removed']}")
        print(f"  Empty rows removed:   {stats['empty_removed']}")
        print(f"  Station names merged: {stats['stations_merged']}")
        print(f"  Output rows:          {stats['output_rows']}")
        print(f"\nData quality:")
        print(f"  Unique stations:      {output_df['STATION_LOCATION'].nunique()}")
        
        # Calculate completeness
        data_cols = [c for c in output_df.columns if c not in ['STATION_LOCATION', 'SITE', 'YEARS']]
        completeness = (output_df[data_cols] != '').sum().sum() / (len(output_df) * len(data_cols)) * 100
        print(f"  Data completeness:    {completeness:.1f}%")
        
        # Save if output path provided
        if output_csv:
            output_df.to_csv(output_csv, index=False)
            print(f"\n✓ Saved cleaned data to: {output_csv}")
        
        print(f"{'='*80}\n")
        
        return output_df
    
    def extract_data_row(self, row, station, site, years, data_start_idx):
        """Extract and clean a single data row."""
        data = {
            'STATION_LOCATION': station,
            'SITE': site,
            'YEARS': years
        }
        
        # Extract numeric columns
        col_names = ['NUM_SAMPLES', 'MIN', 'MAX', 'AVG',
                    'P10', 'P20', 'P30', 'P40', 'P50',
                    'P60', 'P70', 'P80', 'P90', 'PAGE']
        
        for i, col_name in enumerate(col_names):
            col_idx = data_start_idx + i
            if col_idx < len(row):
                value = self.clean_numeric(row.iloc[col_idx])
                data[col_name] = value
            else:
                data[col_name] = ''
        
        return data


def main():
    """Main execution."""
    import os
    
    print("\n" + "="*80)
    print("HISTORICAL AIR POLLUTION DATA - PATTERN-BASED CLEANER")
    print("="*80 + "\n")
    
    # Get file paths
    input_file = input("Enter input CSV path [default=Tables/Table_1_Pages_1-8.csv]: ").strip()
    if not input_file:
        input_file = "Tables/Table_1_Pages_1-8.csv"
    
    output_file = input("Enter output CSV path [default=Tables/Table_1_CLEANED.csv]: ").strip()
    if not output_file:
        output_file = "Tables/Table_1_CLEANED.csv"
    
    # Check if output is a directory, and if so, create a filename
    if os.path.isdir(output_file):
        # Extract input filename and create output filename
        input_basename = os.path.basename(input_file)
        input_name = os.path.splitext(input_basename)[0]
        output_file = os.path.join(output_file, f"{input_name}_CLEANED.csv")
        print(f"Output is a directory. Using filename: {output_file}\n")
    
    # Process
    cleaner = IntelligentTableCleaner()
    
    try:
        result_df = cleaner.clean_table(input_file, output_file)
        
        if len(result_df) > 0:
            # Show preview
            print("\nPreview of cleaned data (first 20 rows):")
            print(result_df.head(20).to_string(index=False, max_colwidth=30))
            
            print("\n\nStation summary (top 20):")
            station_counts = result_df['STATION_LOCATION'].value_counts()
            for station, count in station_counts.head(20).items():
                print(f"  {station:45s} : {count:3d} rows")
        else:
            print("\n⚠ No output generated. Please check your input data format.")
            
    except FileNotFoundError:
        print(f"\n✗ Error: File not found: {input_file}")
        print("  Please check the path and try again.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()