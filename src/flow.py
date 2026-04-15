from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from src.agents import AgentState

class Flow:
    def __init__(self, agents: list, memory: bool = False):
        self.agents = agents

        if memory:
            from langgraph.checkpoint.memory import MemorySaver
            self.checkpointer = MemorySaver()
        else:
            self.checkpointer = None
            
        self.graph = self._build_graph()

    def _build_graph(self):
        graph_builder = StateGraph(AgentState)
        node_names = []

        for i, agent in enumerate(self.agents):
            # 1. Create a Unique Name for this agent node
            # If it's a single agent, we can just call it "agent_0"
            name = f"agent_{i}"
            node_names.append(name)
            
            # 2. Add the Agent Node (using the public .node method we created)
            graph_builder.add_node(name, agent.chatbot_node)

            # 3. Handle Tools for this specific agent
            if agent.tools:
                tool_node_name = f"tools_{i}"
                graph_builder.add_node(tool_node_name, ToolNode(agent.tools))
                
                # Setup the ReAct loop for THIS agent
                graph_builder.add_conditional_edges(
                    name, 
                    tools_condition,
                    {
                        "tools": tool_node_name, 
                        "__end__": "__end__" # tools_condition uses this default
                    }
                )
                graph_builder.add_edge(tool_node_name, name)

        # --- ORCHESTRATION (Connecting the nodes) ---

        # Start at the first agent
        graph_builder.add_edge(START, node_names[0])

        # Connect agent_0 -> agent_1 -> agent_2
        for j in range(len(node_names) - 1):
            graph_builder.add_edge(node_names[j], node_names[j+1])

        # The last agent goes to END
        graph_builder.add_edge(node_names[-1], END)

        return graph_builder.compile(checkpointer=self.checkpointer)
    
    def run(self, user_prompt: str, image_data=None, thread_id="default"):
        # Use the first agent to prepare the message format
        initial_state, config = self.agents[0].prepare(user_prompt, image_data, "image/jpeg", thread_id)
        # Re-attach the flow's checkpointer config
        if self.checkpointer:
            config = {"configurable": {"thread_id": thread_id}}
        
        result = self.graph.invoke(initial_state, config)

        return result["messages"][-1].content

    def stream(
        self,
        user_prompt: str,
        image_data: bytes | list[bytes] | None = None,
        mime_type: str | list[str] = "image/jpeg",
        thread_id: str = "default",
    ):

        from langchain_core.messages import AIMessageChunk

        # 1. DELEGATION: Ask the first agent to package the messages
        # Use the public agent.prepare() method we refactored
        initial_state = self.agents[0].prepare(user_prompt, image_data, mime_type)

        # 2. SESSION MGMT: The Flow handles the thread configuration
        config = {"configurable": {"thread_id": thread_id}} if self.checkpointer else {}

        # 3. EXECUTION: Stream from the graph owned by this Flow
        for chunk, _ in self.graph.stream(
            initial_state,
            config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk):
                content = chunk.content
                if content:
                    if isinstance(content, str):
                        yield content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                yield item.get("text", "")