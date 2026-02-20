"""
Trace Processor - Agent Behavior Mining

This module processes MLflow traces from agent interactions and converts them
into event logs suitable for process mining analysis, implementing the event data
model described in:

"Agent Behavior Mining: Generative AI Agent Governance in Business Processes"

The processor transforms granular agent activities—including reasoning traces,
tool usage, and token costs—into standardized process logs compatible with
process mining tools.
"""

import os
import glob
from typing import List
from .log_generator import LogGenerator
from .xes_exporter import XESExporter
import pandas as pd
from datetime import datetime

class TraceProcessor:    
    def __init__(self, base_path: str = "./mlruns"):
        self.base_path = base_path
        
    def find_trace_files(self) -> List[str]:
        """
        Find all traces.json files in the MLflow directory structure.
        
        Returns:
            List of file paths to trace JSON files
        """
        pattern = os.path.join(self.base_path, "**/traces/**/artifacts/traces.json")
        trace_files = glob.glob(pattern, recursive=True)
        
        # Sort files by modification time (newest first) for better processing order
        trace_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        return trace_files
    
    def process_all_traces(self, export_as_json: bool = False, export_as_xes: bool = False):
        """
        Process all trace files found in the MLflow directory.
        
        Args:
            export_as_json: If True, exports as JSON instead of CSV
            export_as_xes: If True, also exports in XES format for process mining tools
        """

        print("🔍 Searching for trace files...")
        trace_files = self.find_trace_files()
        
        if not trace_files:
            print("❌ No trace files found in ./mlruns directory")
            return {"total": 0, "successful": 0, "failed": 0}
        
        print(f"📁 Found {len(trace_files)} trace files")
        
        
        successful_ingestions = 0
        failed_ingestions = 0

        combined_logs = pd.DataFrame()
        
        for i, file_path in enumerate(trace_files, 1):
            print(f"\t📂 Processing trace {i}/{len(trace_files)}: {file_path}")
            
            log_generator = LogGenerator()
            try:
                trace_event_log = log_generator.generate_event_log_df(file_path)
                combined_logs = pd.concat([combined_logs, trace_event_log], ignore_index=True)
                successful_ingestions += 1
            except Exception as e:
                print(f"   ❌ Failed to generate event log for {file_path}: {e}")
                failed_ingestions += 1
                continue
        
        # Sort combined logs by timestamp
        combined_logs.sort_values(by="time:timestamp", inplace=True)
        
        # Generate log files
        self._generate_log_file(combined_logs, "./generated_event_log", json_format=export_as_json)
        
        # Also generate XES if requested
        if export_as_xes:
            self._generate_xes_file(combined_logs, "./generated_event_log")

        print(f"\n📈 Processing Summary:")
        print(f"   📊 Total trace files processed: {len(trace_files)}")
        print(f"   ✅ Successful: {successful_ingestions}")
        print(f"   ❌ Failed: {failed_ingestions}")
        
        if successful_ingestions > 0:
            print(f"\nLog generation process completed successfully!")
        if len(trace_files) == 0:
            print("\nNo trace files found. Make sure you have completed some coffee shop interactions first.")
            print("💡 Go back to step 4 and create some orders to generate trace data.")
        
        return

    def _generate_log_file(self, dataframe: pd.DataFrame, output_path: str, json_format: bool = False):
        """
        Generate a log file from the given DataFrame.
        
        Args:
            dataframe: The DataFrame containing event log data
            output_path: The path to save the generated log file
            json_format: If True, exports as JSON instead of CSV
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]  # UTC timestamp with ms
        filename = f"{timestamp}.eventlog"
        if json_format:
            filename += ".json"
        else:
            filename += ".csv"

        file_path = os.path.join(output_path, filename)
        
        try:
            if json_format:
                dataframe.to_json(file_path, orient="index")
            else:
                dataframe.to_csv(file_path, index=False)
            print(f"\n✅ CSV/JSON log file generated at {file_path}")
        except Exception as e:
            print(f"\n❌ Failed to generate log file at {file_path}: {e}")
    
    def _generate_xes_file(self, dataframe: pd.DataFrame, output_path: str):
        """
        Generate a XES file from the given DataFrame.
        
        Args:
            dataframe: The DataFrame containing event log data
            output_path: The path to save the generated XES file
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]  # UTC timestamp with ms
        filename = f"{timestamp}.eventlog.xes"
        file_path = os.path.join(output_path, filename)
        
        try:
            exporter = XESExporter()
            exporter.export_to_xes(dataframe, file_path)
            print(f"✅ XES log file generated at {file_path}")
            print(f"   This file can be imported into ProM, Disco, pm4py, and other process mining tools")
        except Exception as e:
            print(f"❌ Failed to generate XES file at {file_path}: {e}")
