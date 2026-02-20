"""
Coffee Shop Multi-Agent System

This module implements the multi-agent Order-to-Cash coffee shop scenario
described in the research paper:
"Agent Behavior Mining: Generative AI Agent Governance in Business Processes"

The system demonstrates how multiple GenAI agents collaborate to handle
customer orders while capturing detailed behavioral traces for process mining analysis.
"""

from langgraph_swarm import create_swarm
from langgraph.checkpoint.memory import InMemorySaver
import mlflow
import uuid
import json
import html
from collections import defaultdict
import ipywidgets as widgets
import os
from IPython.display import display, clear_output, HTML
from langchain.chat_models import init_chat_model

from .agents import (
    MENU, inventory_manager,
    create_order_agent, create_inventory_agent, 
    create_barista_agent, create_customer_service_agent
)
from .styles import ENHANCED_CSS

# CONFIGURE LLM AND REQUIRED CREDENTIALS HERE
# SEE ALSO: https://github.com/langchain-ai/langgraph/blob/a10a66cbd151c92f89d6476fb70e5e405ce50b98/docs/docs/snippets/chat_model_tabs.md
# DO NOT FORGET TO ALSO RUN THE RESPECTIVE `pip install -U "langchain[PROVIDER_NAME]"` COMMAND AS STATED IN THE LINKED DOCUMENTATION
chat_llm = init_chat_model("MODEL_NAME")


mlflow.langchain.autolog()
mlflow_experiment_name = "lg-coffee-mas"
if not mlflow.get_experiment_by_name(mlflow_experiment_name):
    mlflow.create_experiment(mlflow_experiment_name)
mlflow.set_experiment(mlflow_experiment_name)


class CoffeeShop():
    def __init__(self):
        self.agent_definitions = defaultdict(str)
        self.traces_of_latest_conversations = []
        self.verbose_mode = True  # Default to verbose mode
        
        # Agent configuration with icons and colors
        self.agent_config = {
            'order_agent': {
                'icon': '📝',
                'name': 'Order Agent',
                'color': '#2196F3',  # Blue
                'bg_color': '#E3F2FD'
            },
            'inventory_agent': {
                'icon': '📦',
                'name': 'Inventory Agent', 
                'color': '#FF9800',  # Orange
                'bg_color': '#FFF3E0'
            },
            'barista_agent': {
                'icon': '☕',
                'name': 'Barista Agent',
                'color': '#8BC34A',  # Green
                'bg_color': '#F1F8E9'
            },
            'customer_service_agent': {
                'icon': '💬',
                'name': 'Customer Service',
                'color': '#E91E63',  # Pink
                'bg_color': '#FCE4EC'
            },
            'user': {
                'icon': '👤',
                'name': 'You',
                'color': '#424242',  # Dark Gray
                'bg_color': '#F5F5F5'
            }
        }

    
    def set_agent_definition(self, agent, definition):
        """Set or update the definition for a specific agent before starting the shop"""
        self.agent_definitions[agent] = definition


    def open_shop(self):
        """Start the coffee shop application after potentially updating agent definitions"""

        inventory_manager.reset()

        # CREATE AGENTS
        order_agent = create_order_agent(chat_llm, self.agent_definitions.get('order_agent', None))
        inventory_agent = create_inventory_agent(chat_llm, self.agent_definitions.get('inventory_agent', None))
        barista_agent = create_barista_agent(chat_llm, self.agent_definitions.get('barista_agent', None))
        customer_service_agent = create_customer_service_agent(chat_llm, self.agent_definitions.get('customer_service_agent', None))

        # CREATE COFFEE SHOP SWARM
        # https://github.com/langchain-ai/langgraph-swarm-py

        checkpointer = InMemorySaver()

        workflow = create_swarm(
            agents=[order_agent, inventory_agent, barista_agent, customer_service_agent],
            default_active_agent="order_agent",
        )
        self.app = workflow.compile(checkpointer=checkpointer)


    def _get_config(self, thread_id):
        return {"configurable": {"thread_id": thread_id}}


    def _format_content_for_display(self, content):
        try:
            parsed_json = json.loads(content)
            formatted_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            return f'<div class="tool-output"><div class="tool-output-label">Output:</div><pre class="tool-output-code">{html.escape(formatted_json)}</pre></div>'
        except (json.JSONDecodeError, TypeError):
            pass
                    
        # Regular string - escape HTML
        return html.escape(content)

    def _should_show_message_in_silent_mode(self, agent_name, content):
        """Determine if a message should be shown in silent mode"""
        
        user_facing_agents = ['order_agent', 'customer_service_agent', 'barista_agent']
        if agent_name in self.agent_config.keys():
            return True
        
        return False
        

    def _format_message_bubble(self, agent_name, content, is_user=False, is_important=True):
        """Format a message as a chat bubble with agent-specific styling"""
        
        # Get agent configuration
        if is_user:
            config = self.agent_config.get('user')
        else:
            config = self.agent_config.get(agent_name, {
                'icon': '🤖',
                'name': "Uses tool: " + agent_name.replace('_', ' ').title(),
                'color': '#666666',
                'bg_color': '#F0F0F0'
            })
        
        # Format the content for better display - this is the key improvement
        formatted_content = self._format_content_for_display(content)
        
        # Create the HTML for the message bubble
        bubble_html = f"""
        <div style="
            margin: 10px 0;
            padding: 0;
            display: flex;
            align-items: flex-start;
            {'justify-content: flex-end;' if is_user else 'justify-content: flex-start;'}
        " class="chat-bubble {'chat-verbose-message' if not is_important else ''}" >
            <div style="
                max-width: 70%;
                background-color: {config['bg_color']};
                background: linear-gradient(135deg, {config['bg_color']}44, {config['bg_color']}ff);
                border: 2px solid {config['color']};
                border-radius: 15px;
                padding: 12px 16px;
                margin: 0 10px;
                position: relative;
                {'order: 1;' if is_user else ''}
            ">
                <div style="
                    font-weight: bold;
                    color: {config['color']};
                    font-size: 12px;
                    margin-bottom: 5px;
                    display: flex;
                    align-items: center;
                    gap: 5px;
                ">
                    <span style="font-size: 16px;">{config['icon']}</span>
                    {config['name']}
                </div>
                <div style="
                    color: #333;
                    line-height: 1.4;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                ">{formatted_content}</div>
            </div>
        </div>
        """
        
        return bubble_html

    def _auto_scroll_to_bottom(self):
        """Auto-scroll the chat output to the bottom to show the latest message"""
        scroll_script = """
        <script>
        (function() {
            // Find the chat output widget
            var outputs = document.querySelectorAll('.widget-output, .jp-OutputArea-output');
            for (var i = 0; i < outputs.length; i++) {
                var output = outputs[i];
                if (output.style.height === '500px' || output.classList.contains('chat-output')) {
                    // Scroll to bottom with smooth behavior
                    output.scrollTop = output.scrollHeight;
                    break;
                }
            }
            
            // Alternative method: find by chat-output class
            var chatOutputs = document.querySelectorAll('.chat-output');
            chatOutputs.forEach(function(element) {
                element.scrollTop = element.scrollHeight;
            });
            
            // Alternative method: find parent container with overflow auto
            var scrollContainers = document.querySelectorAll('[style*="overflow: auto"], [style*="overflow:auto"]');
            scrollContainers.forEach(function(container) {
                if (container.style.height === '500px') {
                    container.scrollTop = container.scrollHeight;
                }
            });
        })();
        </script>
        """
        display(HTML(scroll_script))

    def _stream_to_output(self, stream, output_widget):
        """Stream conversation to an output widget with enhanced bubble styling"""
        seen_messages = set()
        
        with output_widget:
            for ns, update in stream:
                for node, node_updates in update.items():
                    if node_updates is None:
                        continue

                    if isinstance(node_updates, (dict, tuple)):
                        node_updates_list = [node_updates]
                    elif isinstance(node_updates, list):
                        node_updates_list = node_updates
                    else:
                        raise ValueError(node_updates)

                    for node_updates in node_updates_list:
                        if isinstance(node_updates, tuple):
                            continue
                        messages_key = next(
                            (k for k in node_updates.keys() if "messages" in k), None
                        )
                        if messages_key is not None:
                            message = node_updates[messages_key][-1]
                            # Create a unique identifier for the message
                            message_id = f"{getattr(message, 'content', str(message))}_{getattr(message, 'name', 'unknown')}"
                            
                            # Only display if we haven't seen this message before
                            if message_id not in seen_messages:
                                seen_messages.add(message_id)

                                # Extract agent name and content
                                agent_name = getattr(message, 'name', 'unknown')
                                content = getattr(message, 'content', str(message))
                                
                                
                                is_important = self._should_show_message_in_silent_mode(agent_name, content)
                                # Display the message bubble
                                bubble_html = self._format_message_bubble(agent_name, content, is_user=False, is_important=is_important)
                                display(HTML(bubble_html))
                                # Auto-scroll to bottom after displaying message
                                self._auto_scroll_to_bottom()
                        else:
                            pass

    def _set_processing_status(self, is_processing=True):
        """Update the status indicator to show processing or ready state"""
        if hasattr(self, 'status_indicator'):
            if is_processing:
                self.status_indicator.value = "⏳ Agents are working on your request..."
                # Disable input controls during processing
                self.text_input.disabled = True
                self.send_button.disabled = True
                if hasattr(self, 'restock_button'):
                    self.restock_button.disabled = True
                for button in self.scenario_buttons.children:
                    button.disabled = True
            else:
                self.status_indicator.value = ""
                # Re-enable input controls
                self.text_input.disabled = False
                self.send_button.disabled = False
                self.text_input.focus()
                if hasattr(self, 'restock_button'):
                    self.restock_button.disabled = False
                for button in self.scenario_buttons.children:
                    button.disabled = False

    def continue_conversation_interactive(self, thread_id, prompt, output_widget):
        """Continue conversation with output directed to a widget with enhanced styling"""
        config = self._get_config(thread_id)
        
        # Set processing status
        self._set_processing_status(True)
        
        # Add user message to output with bubble styling
        with output_widget:
            user_bubble = self._format_message_bubble('user', prompt, is_user=True, is_important=True)
            display(HTML(user_bubble))
                    
        try:
            # Stream agent responses to output widget with enhanced styling
            self._stream_to_output(
                self.app.stream(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ]
                    },
                    config,
                    subgraphs=True,
                ),
                output_widget
            )
            trace_id = mlflow.get_last_active_trace_id()
            self.traces_of_latest_conversations.append(trace_id)
            
            with output_widget:
                # Add trace ID in a subtle way
                display(HTML(f"""
                <div style="
                    font-size: 10px;
                    color: #999;
                    text-align: center;
                    margin: 10px 0;
                    padding: 5px;
                    border-top: 1px solid #eee;
                ">
                    Trace ID: {trace_id}
                </div>
                """))
                
        finally:
            # Always reset to ready status when done
            self._set_processing_status(False)

    def _inject_enhanced_css(self):
        """Inject enhanced CSS for the chat interface"""
        css_style = f"<style>\n{ENHANCED_CSS}\n</style>"
        return widgets.HTML(css_style)

    def create_interactive_interface(self, success_only=False):
        """Create an enhanced interactive widget interface for the coffee shop"""
        # Initialize conversation state
        self.current_thread_id = None
        
        # Inject enhanced CSS
        css_widget = self._inject_enhanced_css()

        # ######################
        # Output area

        self.output = widgets.Output()
        self.output.add_class('chat-output')

        # ######################
        # Input area

        self.text_input = widgets.Text(
            value='',
            placeholder='Type your message to the coffee shop here...',
            description='Your Message:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%', height='35px')
        )
        self.text_input.add_class('default-input')
        self.text_input.on_submit(self._on_text_submit)
        
        self.send_button = widgets.Button(
            description='Send 📤',
            button_style='primary',
            layout=widgets.Layout(width="120px", height='35px'),
            tooltip='Send your message'
        )
        self.send_button.add_class('default-button')
        self.send_button.on_click(self._on_send_button_clicked)

        # Create enhanced status indicator
        self.status_indicator = widgets.HTML(
            value=""
        )
        self.status_indicator.add_class('status-indicator')

        input_line = widgets.HBox([
            self.text_input, 
            self.send_button, 
        ])
        input_line.add_class('input-line')

        input_area = widgets.HBox([
            input_line,
            self.status_indicator,
        ], layout=widgets.Layout(justify_content='flex-start', align_items='center'))

        input_area.add_class('input-area')

        # ######################
        # Chat Controls area
        
        controls_header = widgets.HTML("""
        <div>
            <h4 style="margin: 0; color: #007bff;">Chat Control</h4>
            <p style="margin: 0; color: #6c757d;">Use the buttons below to overall control the chat.</p>
        </div>
        """)

        self.new_conversation_button = widgets.Button(
            description='🆕 New Chat',
            button_style='info',
            tooltip='Start a new conversation'
        )
        self.new_conversation_button.add_class('default-button')
        self.new_conversation_button.on_click(self._on_new_conversation_clicked)
        
        # Create verbose/silent mode toggle
        self.verbose_toggle = widgets.ToggleButton(
            value=self.verbose_mode,
            description='🔊 Verbose: On',
            disabled=False,
            button_style='info',
            tooltip='Toggle between verbose (show all messages) and silent (hide tool calls) modes'
        )
        self.verbose_toggle.add_class('default-button')
        self.verbose_toggle.observe(self._on_verbose_toggle_changed, names='value')
        
        # Create enhanced restock button
        self.restock_button = widgets.Button(
            description='🔄 Restock All Items',
            button_style='success',
            tooltip='Restock all items to full inventory'
        )
        self.restock_button.add_class('default-button')
        self.restock_button.on_click(self._on_restock_clicked)
       
        controls_buttons = widgets.HBox([
            self.new_conversation_button,
            self.verbose_toggle,
            self.restock_button
        ])
        controls_buttons.add_class('button-group')
       
        controls_area = widgets.VBox([
            controls_header,
            controls_buttons
        ])

        controls_area.add_class('scenario-area')


        # ######################
        # Scenario area

        scenario_header = widgets.HTML("""
        <div>
            <h4 style="margin: 0; color: #007bff;">Quick Start Scenarios</h4>
            <p style="margin: 0; color: #6c757d;">These buttons will initiate a new conversation using a predefined message.</p>
        </div>
        """)

        # Create enhanced preset scenario buttons
        scenario_buttons = []
        if success_only:
            scenario_buttons.append(widgets.Button(
                description='🛍️ Successful Order',
                button_style='success',
                tooltip='Order 2 lattes and a croissant'
            ))
        else:
            scenario_buttons.append(widgets.Button(
                description='❓ Menu Issue',
                button_style='warning', 
                tooltip='Order item not on menu'
            ))
            scenario_buttons.append(widgets.Button(
                description='📦 Inventory Issue',
                button_style='danger',
                tooltip='Order item out of stock'
            ))
            scenario_buttons.append(widgets.Button(
                description='😞 Complaint',
                button_style='info',
                tooltip='Complain about a drink'
            ))
       
        for button in scenario_buttons:
            button.add_class('scenario-button')
            button.add_class('default-button')
            
        self.scenario_buttons = widgets.HBox(scenario_buttons)
        self.scenario_buttons.add_class('button-group')

        # Set up scenario button handlers
        for i, button in enumerate(self.scenario_buttons.children):
            button.on_click(lambda b, scenario=i: self._on_scenario_clicked(b, scenario))
        

        scenario_area = widgets.VBox([
            scenario_header,
            self.scenario_buttons,
        ])
        
        scenario_area.add_class('scenario-area')


        # ######################
        # Combine all parts into main interface

        controls = widgets.HBox([
            scenario_area,
            controls_area,
        ])
        controls.add_class('controls-container')

        chat_area = widgets.VBox([
            self.output,
            input_area,
        ])
        chat_area.add_class('chat-area')

        interface = widgets.VBox([
            css_widget,  # Inject CSS
            widgets.HTML('<div style="margin: 5px 0;"></div>'),  # Spacer
            chat_area,
            controls
        ])
        
        interface.add_class('chat-container')
        
        # Start first conversation
        self._start_new_conversation()
        
        return interface
    
    def _on_send_button_clicked(self, button):
        """Handle send button click"""
        message = self.text_input.value.strip()
        if message:
            self.continue_conversation_interactive(self.current_thread_id, message, self.output)
            self.text_input.value = ''
    
    def _on_text_submit(self, text_widget):
        """Handle text input submission (Enter key)"""
        self._on_send_button_clicked(None)
    
    def _on_new_conversation_clicked(self, button):
        """Handle new conversation button click"""
        print("These are the trace IDs of the latest conversations in this session:")
        for trace_id in self.traces_of_latest_conversations:
            print(f"- {trace_id}")
        self.traces_of_latest_conversations = []
        
        self._start_new_conversation()
    
    def _start_new_conversation(self):
        """Start a new conversation thread"""
        self.current_thread_id = str(uuid.uuid4())

        with self.output:
            clear_output()
            welcome_html = """
            <div style="
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 15px;
            ">
                <h3 style="margin: 0;">🆕 New Conversation Started!</h3>
            </div>
            """
            display(HTML(welcome_html))
    
    def _on_restock_clicked(self, button):
        """Handle restock button click"""
        # Use the inventory_manager's reset method
        inventory_manager.reset()
        
        # Show enhanced confirmation message in output
        with self.output:
            restock_html = """
            <div style="
                background: linear-gradient(45deg, #d4edda, #a3d977);
                border: 1px solid #28a745;
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                text-align: center;
            ">
                <h4 style="margin: 0 0 10px 0; color: #155724;">🔄 Inventory Successfully Restocked!</h4>
            </div>
            """
            display(HTML(restock_html))
            
        self.display_current_inventory()
    
    def display_current_inventory(self):
        with self.output:            
            # Display current inventory in a nice format
            inventory_html = '<div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">'
            inventory_html += '<h5 style="margin: 0 0 10px 0; color: #495057;">📦 Current Inventory Levels:</h5>'
            inventory_html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">'
            
            for item_key, item in inventory_manager.inventory.items():
                inventory_html += f"""
                <div style="
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    border-left: 3px solid #007bff;
                ">
                    <strong>{item.name}</strong><br>
                    <span style="color: #28a745;">Stock: {item.stock} units</span><br>
                    <span style="color: #6c757d;">${item.price:.2f}</span>
                </div>
                """
            
            inventory_html += '</div></div>'
            display(HTML(inventory_html))

    def _on_scenario_clicked(self, button, scenario):
        """Handle preset scenario button clicks"""
        self._start_new_conversation()
        
        scenarios = []

        if len(self.scenario_buttons.children) == 1:
            # Only success scenario button is present
            scenarios = [
                "I'd like to order 2 large lattes with almond milk and 1 croissant please"
            ]
        else:
            # All scenario buttons are present
            scenarios = [
                "I want 1 croissant and 1 piece of cheesecake",
                "Can I get 2 muffins please?",
                "I'm not happy with my latte, it tastes bitter and wrong"
            ]
            # Modify inventory for the inventory issue scenario
            if scenario == 1:
                inventory_manager.inventory['muffin'].stock = 0
                with self.output:
                    restock_html = """
                    <div class="chat-notification">
                        <h4>All muffins vanished from the inventory!</h4>
                    </div>
                    """
                    display(HTML(restock_html))
                self.display_current_inventory()
            else:
                if inventory_manager.inventory['muffin'].stock == 0:
                    with self.output:
                        restock_html = """
                        <div class="chat-notification">
                            <h4>Fresh muffins arrived!</h4>
                        </div>
                        """
                        display(HTML(restock_html))
                    inventory_manager.inventory['muffin'].stock = 12

        self.text_input.value = scenarios[scenario]
        self._on_send_button_clicked(None)

    def _on_verbose_toggle_changed(self, change):
        """Handle verbose mode toggle changes"""
        self.verbose_mode = change['new']
        
        # Update toggle description and button style based on mode
        if self.verbose_mode:
            self.output.remove_class('chat-silent-mode')
            self.verbose_toggle.description = '🔊 Verbose: On'
            self.verbose_toggle.button_style = 'info'
        else:
            self.verbose_toggle.description = '🔇 Verbose: Off'
            self.verbose_toggle.button_style = 'warning'
            self.output.add_class('chat-silent-mode')
