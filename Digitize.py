"""
Automated Table Reconstruction from OCR Results
Uses detected column structures to rebuild original tables
"""

import pandas as pd
import numpy as np
import os
import re
from typing import List, Dict, Tuple


def load_column_structure(column_file: str) -> List[int]:
    """
    Load column line positions from a column structure file.
    
    Args:
        column_file: Path to *_columns.txt file
        
    Returns:
        List of x-positions for column lines
    """
    column_lines = []
    
    if not os.path.exists(column_file):
        return column_lines
    
    with open(column_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('line_'):
                # Parse: "line_1: 245"
                parts = line.split(':')
                if len(parts) == 2:
                    x_pos = int(parts[1].strip())
                    column_lines.append(x_pos)
    
    return sorted(column_lines)


def create_column_definitions(column_lines: List[int], padding: int = 60) -> List[Tuple[str, int, int]]:
    """
    Create column definitions from detected column lines.
    
    Args:
        column_lines: List of x-positions for column separators
        padding: Padding added during preprocessing
        
    Returns:
        List of tuples: (column_name, x_min, x_max)
    """
    if not column_lines:
        # If no columns detected, return single wide column
        return [('Column_0', 0, 10000)]
    
    # Adjust for padding
    adjusted_lines = [x + padding for x in column_lines]
    
    col_definitions = []
    
    # First column: from 0 to first line
    col_definitions.append(('Column_0', 0, adjusted_lines[0]))
    
    # Middle columns: between consecutive lines
    for i in range(len(adjusted_lines) - 1):
        col_name = f'Column_{i+1}'
        col_definitions.append((col_name, adjusted_lines[i], adjusted_lines[i+1]))
    
    # Last column: from last line to infinity
    col_name = f'Column_{len(adjusted_lines)}'
    col_definitions.append((col_name, adjusted_lines[-1], 10000))
    
    return col_definitions


def assign_column(x_pos: int, col_definitions: List[Tuple[str, int, int]]) -> str:
    """
    Assign a column name based on x-position.
    """
    for col_name, x_min, x_max in col_definitions:
        if x_min <= x_pos < x_max:
            return col_name
    return None


def load_table_groups(structure_file: str) -> List[Dict]:
    """
    Parse the table structure report to get table groups.
    
    Args:
        structure_file: Path to table_structure_report.txt
        
    Returns:
        List of table group dictionaries
    """
    if not os.path.exists(structure_file):
        print(f"Warning: {structure_file} not found")
        return []
    
    groups = []
    current_group = None
    
    with open(structure_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Look for "Table Group X:"
            if line.startswith('Table Group'):
                if current_group:
                    groups.append(current_group)
                current_group = {}
                
            # Look for "Pages: X to Y"
            elif line.startswith('Pages:') and current_group is not None:
                # Extract page range
                match = re.search(r'Pages:\s+(\d+)\s+to\s+(\d+)', line)
                if match:
                    current_group['start_page'] = int(match.group(1))
                    current_group['end_page'] = int(match.group(2))
                else:
                    # Single page
                    match = re.search(r'Pages:\s+(\d+)', line)
                    if match:
                        current_group['start_page'] = int(match.group(1))
                        current_group['end_page'] = int(match.group(1))
            
            # Look for "Page numbers: [...]"
            elif line.startswith('Page numbers:') and current_group is not None:
                # Extract list of page numbers
                match = re.search(r'Page numbers:\s+\[(.*?)\]', line)
                if match:
                    pages_str = match.group(1)
                    current_group['pages'] = [int(p.strip()) for p in pages_str.split(',')]
            
            # Look for "Number of columns:"
            elif line.startswith('Number of columns:') and current_group is not None:
                match = re.search(r'Number of columns:\s+(\d+)', line)
                if match:
                    current_group['num_columns'] = int(match.group(1))
    
    # Add last group
    if current_group:
        groups.append(current_group)
    
    return groups


def reconstruct_table(df: pd.DataFrame, page_num: int, col_definitions: List[Tuple[str, int, int]], 
                      row_tolerance: int = 32) -> pd.DataFrame:
    """
    Reconstruct a table from OCR data for a specific page.
    
    Args:
        df: Full OCR dataframe
        page_num: Page number to process
        col_definitions: Column definitions for this page
        row_tolerance: Vertical tolerance for grouping items into rows (pixels)
        
    Returns:
        DataFrame representing the reconstructed table
    """
    df_page = df[df['page_num'] == page_num].copy()
    
    if df_page.empty:
        return pd.DataFrame()
    
    # Assign columns based on x-position
    df_page['column'] = df_page['left'].apply(lambda x: assign_column(x, col_definitions))
    
    # Group by row position with tolerance
    df_page['row_id'] = df_page['top'].apply(lambda x: round(x / row_tolerance) * row_tolerance)
    unique_rows = sorted(df_page['row_id'].unique())
    
    # Get column names
    columns = [col[0] for col in col_definitions]
    result_data = []
    
    # Process each row
    for row_id in unique_rows:
        row_data = df_page[df_page['row_id'] == row_id].sort_values('left')
        row_dict = {col: '' for col in columns}
        
        for _, item in row_data.iterrows():
            col = item['column']
            if col and col in row_dict:
                if row_dict[col]:
                    row_dict[col] += ' ' + str(item['text'])
                else:
                    row_dict[col] = str(item['text'])
        
        # Only add row if it has content
        if any(row_dict.values()):
            row_dict['Page'] = page_num
            result_data.append(row_dict)
    
    return pd.DataFrame(result_data)


def remove_header_rows(df: pd.DataFrame, header_keywords: List[str] = None) -> pd.DataFrame:
    """
    Remove header/title rows from the reconstructed table.
    
    Args:
        df: DataFrame to filter
        header_keywords: List of keywords to identify header rows
        
    Returns:
        Filtered DataFrame
    """
    if header_keywords is None:
        header_keywords = [
            'STATION LOCATION', 'SITE', 'YEARS', 'SAMPLES', 
            'MICROGRAMS', 'FREQUENCY', 'DISTRIBUTION', 'PERCENT',
            'MIN', 'MAX', 'AVG', 'TABLE', 'SUSPENDED', 'PARTICULATE',
            'URBAN', 'STATIONS', 'MATTER'
        ]
    
    def is_header_row(row):
        row_text = ' '.join([str(v) for v in row.values if v])
        return any(keyword in row_text.upper() for keyword in header_keywords)
    
    filtered_df = df[~df.apply(is_header_row, axis=1)]
    return filtered_df.reset_index(drop=True)


def reconstruct_all_tables(ocr_csv: str, processed_dir: str, output_dir: str = "Tables"):
    """
    Reconstruct all tables from OCR results using detected column structures.
    
    Args:
        ocr_csv: Path to combined_ocr_results.csv
        processed_dir: Directory containing column structure files
        output_dir: Directory to save reconstructed tables
    """
    # Load OCR data
    print("=" * 70)
    print("AUTOMATED TABLE RECONSTRUCTION")
    print("=" * 70)
    print(f"\nLoading OCR results from: {ocr_csv}")
    df = pd.read_csv(ocr_csv)
    print(f"Loaded {len(df)} OCR entries from {df['page_num'].nunique()} pages")

    # --- CONFIDENCE FILTER ---
    if 'conf' in df.columns:
        before = len(df)
        df = df[df['conf'] >= 70].copy()
        removed = before - len(df)
        print(f"Filtered out low-confidence text (<70): {removed} removed, {len(df)} remaining")
    else:
        print("Warning: No 'conf' column found — skipping confidence filtering.")


    
    # Load table groups
    structure_file = os.path.join(processed_dir, 'table_structure_report.txt')
    table_groups = load_table_groups(structure_file)
    
    if not table_groups:
        print("\nWarning: No table groups found. Processing all pages individually.")
        table_groups = [{
            'start_page': p,
            'end_page': p,
            'pages': [p],
            'num_columns': 0
        } for p in sorted(df['page_num'].unique())]
    
    print(f"\nFound {len(table_groups)} table group(s)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_tables = []
    
    # Process each table group
    for group_idx, group in enumerate(table_groups, 1):
        print(f"\n{'='*70}")
        print(f"Processing Table Group {group_idx}")
        print(f"{'='*70}")
        print(f"Pages: {group['start_page']} to {group['end_page']}")
        print(f"Page numbers: {group.get('pages', [])}")
        print(f"Expected columns: {group.get('num_columns', 'Unknown')}")
        
        group_data = []
        
        # Process each page in the group
        for page_num in group.get('pages', []):
            print(f"\n  Processing page {page_num}...")
            
            # Find column structure file for this page
            # Look for files matching pattern: *-{page_num:03d}_columns.txt
            column_files = [
                f for f in os.listdir(processed_dir)
                if f.endswith('_columns.txt') and f'-{page_num:03d}_' in f
            ]
            
            if column_files:
                column_file = os.path.join(processed_dir, column_files[0])
                column_lines = load_column_structure(column_file)
                print(f"    Loaded {len(column_lines)} column lines from {column_files[0]}")
            else:
                # Try without zero-padding
                column_files = [
                    f for f in os.listdir(processed_dir)
                    if f.endswith('_columns.txt') and f'-{page_num}_' in f
                ]
                if column_files:
                    column_file = os.path.join(processed_dir, column_files[0])
                    column_lines = load_column_structure(column_file)
                    print(f"    Loaded {len(column_lines)} column lines from {column_files[0]}")
                else:
                    print(f"    Warning: No column file found for page {page_num}")
                    column_lines = []
            
            # Create column definitions
            col_definitions = create_column_definitions(column_lines)
            print(f"    Created {len(col_definitions)} column definitions")
            
            # Reconstruct table for this page
            page_table = reconstruct_table(df, page_num, col_definitions)
            
            if not page_table.empty:
                group_data.append(page_table)
                print(f"    Extracted {len(page_table)} rows")
            else:
                print(f"    No data extracted")
        
        # Combine all pages in this group
        if group_data:
            group_table = pd.concat(group_data, ignore_index=True)
            
            # Remove header rows
            group_table = remove_header_rows(group_table)
            
            # Add separator row indicating table group
            separator_row = {col: '' for col in group_table.columns}
            separator_row[group_table.columns[0]] = f"=== TABLE {group_idx}: Pages {group['start_page']}-{group['end_page']} ==="
            separator_df = pd.DataFrame([separator_row])
            
            # Add to all tables
            all_tables.append(separator_df)
            all_tables.append(group_table)
            
            # Save individual table
            table_file = os.path.join(output_dir, f"Table_{group_idx}_Pages_{group['start_page']}-{group['end_page']}.csv")
            group_table.to_csv(table_file, index=False)
            print(f"\n  ✓ Saved Table {group_idx}: {len(group_table)} rows → {table_file}")
        else:
            print(f"\n  ⚠ No data extracted for Table Group {group_idx}")
    
    # Combine all tables into one file
    if all_tables:
        combined_table = pd.concat(all_tables, ignore_index=True)
        combined_file = os.path.join(output_dir, "All_Tables_Combined.csv")
        combined_table.to_csv(combined_file, index=False)
        
        print("\n" + "=" * 70)
        print("RECONSTRUCTION COMPLETE!")
        print("=" * 70)
        print(f"Total tables created: {len(table_groups)}")
        print(f"Total rows extracted: {len(combined_table)}")
        print(f"\nOutput files:")
        print(f"  • Combined tables: {combined_file}")
        print(f"  • Individual tables: {output_dir}/Table_*.csv")
        print("=" * 70)
        
        return combined_table
    else:
        print("\n⚠ No tables could be reconstructed")
        return pd.DataFrame()


def main():
    """Main execution function."""
    
    print("\nAUTOMATED TABLE RECONSTRUCTION FROM OCR")
    print("Uses detected column structures to rebuild original tables\n")
    
    # Get paths
    ocr_csv = input("Enter path to combined_ocr_results.csv [default=Processed/combined_ocr_results.csv]: ").strip()
    if not ocr_csv:
        ocr_csv = "Processed/combined_ocr_results.csv"
    
    if not os.path.exists(ocr_csv):
        print(f"Error: File not found: {ocr_csv}")
        return
    
    processed_dir = input("Enter processed files directory [default=Processed]: ").strip()
    if not processed_dir:
        processed_dir = "Processed"
    
    if not os.path.exists(processed_dir):
        print(f"Error: Directory not found: {processed_dir}")
        return
    
    output_dir = input("Enter output directory for tables [default=Tables]: ").strip()
    if not output_dir:
        output_dir = "Tables"
    
    # Reconstruct tables
    result = reconstruct_all_tables(ocr_csv, processed_dir, output_dir)
    
    if not result.empty:
        print(f"\n✓ Successfully reconstructed tables!")
        print(f"\nFirst few rows of combined table:")
        print(result.head(20).to_string())


if __name__ == "__main__":
    main()