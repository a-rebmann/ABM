"""
XES Exporter - Agent Behavior Mining

This module exports event logs to XES (eXtensible Event Stream) format,
the standard for process mining tools, as described in:

"Agent Behavior Mining: Generative AI Agent Governance in Business Processes"

The exporter implements the XES standard (IEEE 1849-2016) with extensions
for agent-specific attributes aligned with the ABM event data model.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional


class XESExporter:
    """
    Export event logs to XES format with agent-specific extensions.
    
    Supports the standard XES extensions (concept, time, org, lifecycle)
    plus custom extensions for GenAI agent attributes (model, tokens, tools).
    """
    
    def __init__(self):
        self.xes_namespaces = {
            'xes': 'http://www.xes-standard.org/',
        }
        
    def export_to_xes(self, dataframe: pd.DataFrame, output_path: str) -> str:
        """
        Export a pandas DataFrame to XES format.
        
        Args:
            dataframe: DataFrame containing event log data
            output_path: Path to save the XES file
            
        Returns:
            Path to the generated XES file
        """
        # Create root element
        log = ET.Element('log')
        log.set('xes.version', '1849-2016')
        log.set('xes.features', 'nested-attributes')
        
        # Add extensions
        self._add_extensions(log)
        
        # Add global attributes
        self._add_global_attributes(log)
        
        # Add classifiers
        self._add_classifiers(log)
        
        # Group by case_id to create traces
        cases = dataframe.groupby('case_id')
        
        for case_id, case_events in cases:
            trace = self._create_trace(case_id, case_events)
            log.append(trace)
        
        # Convert to string with pretty printing
        xml_string = self._prettify_xml(log)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_string)
        
        return output_path
    
    def _add_extensions(self, log: ET.Element):
        """Add XES extensions for standard and custom attributes"""
        
        # Standard XES extensions
        extensions = [
            ('Concept', 'concept', 'http://www.xes-standard.org/concept.xesext'),
            ('Time', 'time', 'http://www.xes-standard.org/time.xesext'),
            ('Organizational', 'org', 'http://www.xes-standard.org/org.xesext'),
            ('Identity', 'identity', 'http://www.xes-standard.org/identity.xesext'),
        ]
        
        for name, prefix, uri in extensions:
            ext = ET.SubElement(log, 'extension')
            ext.set('name', name)
            ext.set('prefix', prefix)
            ext.set('uri', uri)
        
        # Custom extension for AI/Agent attributes
        ai_ext = ET.SubElement(log, 'extension')
        ai_ext.set('name', 'AI Agent')
        ai_ext.set('prefix', 'ai')
        ai_ext.set('uri', 'http://www.agent-behavior-mining.org/ai.xesext')
    
    def _add_global_attributes(self, log: ET.Element):
        """Add global event and trace attributes"""
        
        # Global trace attributes
        global_trace = ET.SubElement(log, 'global')
        global_trace.set('scope', 'trace')
        
        self._add_string_attribute(global_trace, 'concept:name', 'Agent Interaction')
        
        # Global event attributes
        global_event = ET.SubElement(log, 'global')
        global_event.set('scope', 'event')
        
        self._add_string_attribute(global_event, 'concept:name', 'Activity')
        self._add_string_attribute(global_event, 'concept:instance', 'Activity Instance')
        self._add_string_attribute(global_event, 'org:resource', 'Agent')
        self._add_date_attribute(global_event, 'time:timestamp', '1970-01-01T00:00:00.000Z')
    
    def _add_classifiers(self, log: ET.Element):
        """Add event classifiers"""
        
        # Activity classifier (by concept:name)
        classifier1 = ET.SubElement(log, 'classifier')
        classifier1.set('name', 'Activity')
        classifier1.set('keys', 'concept:name')
        
        # Activity+Agent classifier
        classifier2 = ET.SubElement(log, 'classifier')
        classifier2.set('name', 'Activity and Agent')
        classifier2.set('keys', 'concept:name org:resource')
        
        # Instance classifier (detailed view)
        classifier3 = ET.SubElement(log, 'classifier')
        classifier3.set('name', 'Instance')
        classifier3.set('keys', 'concept:instance')
    
    def _create_trace(self, case_id: str, events: pd.DataFrame) -> ET.Element:
        """Create a trace element for a case"""
        
        trace = ET.Element('trace')
        
        # Add case attributes
        self._add_string_attribute(trace, 'concept:name', str(case_id))
        self._add_string_attribute(trace, 'identity:id', str(case_id))
        
        # Add events
        for _, event_data in events.iterrows():
            event = self._create_event(event_data)
            trace.append(event)
        
        return trace
    
    def _create_event(self, event_data: pd.Series) -> ET.Element:
        """Create an event element from event data"""
        
        event = ET.Element('event')
        
        # Standard attributes
        if pd.notna(event_data.get('concept:name')):
            self._add_string_attribute(event, 'concept:name', str(event_data['concept:name']))
        
        if pd.notna(event_data.get('concept:instance')):
            self._add_string_attribute(event, 'concept:instance', str(event_data['concept:instance']))
        
        if pd.notna(event_data.get('org:resource')):
            self._add_string_attribute(event, 'org:resource', str(event_data['org:resource']))
        
        if pd.notna(event_data.get('identity:id')):
            self._add_string_attribute(event, 'identity:id', str(event_data['identity:id']))
        
        # Timestamps
        if pd.notna(event_data.get('time:timestamp')):
            timestamp = self._format_timestamp(event_data['time:timestamp'])
            self._add_date_attribute(event, 'time:timestamp', timestamp)
        
        if pd.notna(event_data.get('time_finished')):
            timestamp_finished = self._format_timestamp(event_data['time_finished'])
            self._add_date_attribute(event, 'time:finished', timestamp_finished)
        
        # Duration
        if pd.notna(event_data.get('duration')):
            # Convert nanoseconds to milliseconds
            duration_ms = float(event_data['duration']) / 1_000_000
            self._add_float_attribute(event, 'time:duration', duration_ms)
        
        # AI/Agent-specific attributes
        if pd.notna(event_data.get('model')):
            self._add_string_attribute(event, 'ai:model', str(event_data['model']))
        
        if pd.notna(event_data.get('input_tokens')):
            self._add_int_attribute(event, 'ai:input_tokens', int(event_data['input_tokens']))
        
        if pd.notna(event_data.get('response_tokens')):
            self._add_int_attribute(event, 'ai:output_tokens', int(event_data['response_tokens']))
        
        if pd.notna(event_data.get('tool')):
            self._add_string_attribute(event, 'ai:tool', str(event_data['tool']))
        
        if pd.notna(event_data.get('message')):
            self._add_string_attribute(event, 'ai:message', str(event_data['message']))
        
        return event
    
    def _add_string_attribute(self, parent: ET.Element, key: str, value: str):
        """Add a string attribute to an element"""
        attr = ET.SubElement(parent, 'string')
        attr.set('key', key)
        attr.set('value', value)
    
    def _add_date_attribute(self, parent: ET.Element, key: str, value: str):
        """Add a date attribute to an element"""
        attr = ET.SubElement(parent, 'date')
        attr.set('key', key)
        attr.set('value', value)
    
    def _add_int_attribute(self, parent: ET.Element, key: str, value: int):
        """Add an integer attribute to an element"""
        attr = ET.SubElement(parent, 'int')
        attr.set('key', key)
        attr.set('value', str(value))
    
    def _add_float_attribute(self, parent: ET.Element, key: str, value: float):
        """Add a float attribute to an element"""
        attr = ET.SubElement(parent, 'float')
        attr.set('key', key)
        attr.set('value', str(value))
    
    def _format_timestamp(self, timestamp: Any) -> str:
        """
        Format a timestamp to XES-compliant ISO 8601 format.
        
        Args:
            timestamp: Can be string, datetime, or pandas Timestamp
            
        Returns:
            ISO 8601 formatted timestamp string
        """
        if isinstance(timestamp, str):
            # Already a string, ensure proper format
            if 'T' in timestamp:
                # ISO format, ensure it has timezone
                if not (timestamp.endswith('Z') or '+' in timestamp or timestamp.count(':') > 2):
                    timestamp += 'Z'
                return timestamp
            else:
                # Try to parse and reformat
                try:
                    dt = pd.to_datetime(timestamp)
                    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                except:
                    return timestamp + 'Z'
        
        elif isinstance(timestamp, (datetime, pd.Timestamp)):
            return timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        
        else:
            # Assume it's a numeric timestamp (nanoseconds)
            try:
                dt = pd.to_datetime(timestamp, unit='ns')
                return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            except:
                return str(timestamp)
    
    def _prettify_xml(self, elem: ET.Element) -> str:
        """
        Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        
        # Get pretty XML with proper indentation
        pretty = reparsed.toprettyxml(indent='  ', encoding='UTF-8').decode('utf-8')
        
        # Remove extra blank lines
        lines = [line for line in pretty.split('\n') if line.strip()]
        
        return '\n'.join(lines)