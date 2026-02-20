"""
Event Log Generator - Agent Behavior Mining

This module generates standardized event logs from agent traces, implementing
the event data model described in:

"Agent Behavior Mining: Generative AI Agent Governance in Business Processes"

It translates agent activities (LLM calls, tool executions, agent handoffs) into
XES-compatible event attributes for process mining analysis.
"""

import json
import json
import uuid
import pandas as pd


class LogGenerator:
    def generate_event_log_df(self, tracefile_path: str) -> pd.DataFrame:
        self.process_events = []
        self.case_id = None
        self.spans = None
        self.langgraph_root_span = None
        
        try:
            with open(tracefile_path, 'r') as f:
                trace_data = json.load(f)
        except Exception as e:
            raise Exception(f'Error loading trace file {tracefile_path}: {e}')

        if 'spans' in trace_data:
            self.spans = trace_data['spans']
        elif 'data' in trace_data and 'spans' in trace_data['data']:
            self.spans = trace_data['data']['spans']
        else:
            raise Exception('Cannot locate spans in trace data!')

        # This is the root node of the LangGraph trace
        self.langgraph_root_span = [span for span in self.spans if span['name'] == 'LangGraph'][0]
        self.case_id = json.loads(self.langgraph_root_span['attributes']['metadata'])['thread_id']

        self._process_root_span()

        if self._is_agent_span(self.langgraph_root_span):
            self._process_agent_span(self.langgraph_root_span)
        else:
            agent_spans = [span for span in self.spans if span['parent_span_id'] == self.langgraph_root_span['span_id']]
            
            for agent_span in agent_spans:
                self._process_agent_span(agent_span)

        dataframe = pd.DataFrame(self.process_events).sort_values(["time:timestamp"], ascending=True)
        dataframe['time_finished'] = ((dataframe['time:timestamp'] + dataframe['duration'].fillna(0))).apply(lambda t: pd.to_datetime(t).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3])
        dataframe['time:timestamp'] = dataframe['time:timestamp'].apply(lambda t: pd.to_datetime(t).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3])

        return dataframe


    def _get_span_metadata(self, span):
        return json.loads(span['attributes']['metadata'])


    def _is_agent_span(self, span):
        child_spans = [s for s in self.spans if s['parent_span_id'] == span['span_id']]
        for child in child_spans:
            if child['name'].startswith('agent_'):
                grandchild_spans = [s for s in self.spans if s['parent_span_id'] == child['span_id']]
                for grandchild in grandchild_spans:
                    if grandchild['name'].startswith('call_model'):
                        return True

            child_metadata = self._get_span_metadata(child)
            if child_metadata['langgraph_node'] == "tools":
                return True

        return False


    def _process_llm_span(self, span, agent_name):
        call_model_child_spans = [s for s in self.spans if s['parent_span_id'] == span['span_id'] and s['name'].startswith('call_model')]
        if len(call_model_child_spans) != 1:
            print(f'Expected exactly one call_model child span for agent span {span["name"]}, found {len(call_model_child_spans)}:\n{[child["name"] for child in call_model_child_spans]}')
            return

        span = call_model_child_spans[0]

        span_output = json.loads(span['attributes']['mlflow.spanOutputs'])['messages'][0]

        model_name = span_output.get('response_metadata', {}).get('model_name', None)

        response_message = None
        returned_contents = span_output.get('content', [])

        if isinstance(returned_contents, str):
            response_message = returned_contents
        elif isinstance(returned_contents, list):
            for content in returned_contents:
                if content.get('type', None) == 'text':
                    response_message = content.get('text', None)
        
        usage_metadata = span_output.get('usage_metadata', {})

        self.process_events.append({
            'case_id': self.case_id,
            'identity:id': str(uuid.uuid4()),
            'time:timestamp': span['start_time_unix_nano'],
            'time_finished': span['end_time_unix_nano'],
            'duration': span['end_time_unix_nano']-span['start_time_unix_nano'],
            'concept:instance': f'{agent_name} calls llm',
            'concept:name': 'call_llm',
            'org:resource': agent_name,
            'model': model_name,
            'input_tokens': usage_metadata.get('input_tokens', None),
            'response_tokens': usage_metadata.get('output_tokens', None),
            'message': response_message
        })


    def _process_tool_span(self, span, agent_name):
        tool_input = json.loads(span['attributes'].get('mlflow.spanInputs', '[]'))[0]
        tool_name = 'unknown_tool'
        
        if tool_input.get('type', None) == 'tool_call':
            tool_name = tool_input.get('name', 'unknown_tool')

        if tool_name.startswith('transfer_to_'):
            return
        
        self.process_events.append({
            'case_id': self.case_id,
            'identity:id': str(uuid.uuid4()),
            'time:timestamp': span['start_time_unix_nano'],
            'time_finished': span['end_time_unix_nano'],
            'duration': span['end_time_unix_nano']-span['start_time_unix_nano'],
            'concept:name': 'execute_tool',
            'concept:instance': f'{agent_name} uses tool {tool_name}',
            'org:resource': agent_name,
            'tool': tool_name,
        })


    def _process_agent_span(self, agent_span):
        agent_span_id = agent_span['span_id']
        agent_metadata = json.loads(agent_span['attributes']['metadata'])

        agent_name = None

        if agent_span['name'] == 'LangGraph':
            agent_name = 'Agent'
        else:
            agent_name = agent_metadata['langgraph_node']

        if agent_name in ["__start__"]:
            return

        agent_child_spans = [span for span in self.spans if span['parent_span_id'] == agent_span_id]
        
        if not self._is_agent_span(agent_span):
            if (len(agent_child_spans) != 1):
                raise Exception(f'Expected exactly one child span for non-agent span {agent_span["name"]}, found {len(agent_child_spans)}:\n{[child["name"] for child in agent_child_spans]}')
            sub_agent_span = agent_child_spans[0]
            agent_child_spans = [span for span in self.spans if span['parent_span_id'] == sub_agent_span['span_id']]

        for child_span in agent_child_spans:
            if child_span['name'].startswith('agent'):
                self._process_llm_span(child_span, agent_name)
            elif child_span['name'].startswith('tools'):
                self._process_tool_span(child_span, agent_name)


    def _process_root_span(self):
        user_input = None
        span_inputs = json.loads(self.langgraph_root_span['attributes']['mlflow.spanInputs'])['messages']
        for message in span_inputs:
            if message.get('role', "") == "user" or message.get('type', None) == "human":
                user_input = message.get('content', None)

        self.process_events.append({
            'case_id': self.case_id,
            'identity:id': str(uuid.uuid4()),
            'time:timestamp': self.langgraph_root_span['start_time_unix_nano'],
            'concept:instance': 'prompt',
            'concept:name': 'user_prompt',
            'org:resource': 'user',
            'message': user_input,
        })