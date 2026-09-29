import { tool } from '@langchain/core/tools'
import z from 'zod'
import { ChatOpenAI } from '@langchain/openai'
import { config } from 'dotenv'

import { StateGraph, StateSchema, MessagesValue } from '@langchain/langgraph'
import { ToolNode } from '@langchain/langgraph/prebuilt'

// load env
config()

const llm = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4o'
})

const multiply = tool(
  ({ a, b }) => {
    return a * b
  },
  {
    name: 'multiply',
    description: 'Multiply two numbers together',
    schema: z.object({
      a: z.number().describe('first number'),
      b: z.number().describe('second number')
    })
  }
)

const add = tool(
  ({ a, b }) => {
    return a + b
  },
  {
    name: 'add',
    description: 'Add two numbers together',
    schema: z.object({
      a: z.number().describe('first number'),
      b: z.number().describe('second number')
    })
  }
)

const divide = tool(
  ({ a, b }) => {
    return a / b
  },
  {
    name: 'divide',
    description: 'Divide two numbers',
    schema: z.object({
      a: z.number().describe('first number'),
      b: z.number().describe('second number')
    })
  }
)

// Augmenconst
const tools = [add, multiply, divide]
const llmWithTools = llm.bindTools(tools)

// Graph State
const State = new StateSchema({
  messages: MessagesValue
})

// Nodes
const llmCall = async (state) => {
  // LLM decides whether to call a tool or not
  const result = await llmWithTools.invoke([
    {
      role: 'system',
      content:
        'You are a helpful assistant tasked with performing arithmetic on a set of inputs.'
    },
    ...state.messages
  ])
  return {
    messages: [result]
  }
}

const toolNode = new ToolNode(tools)

// Conditional edge function to route to the tool node or end
const shouldContinue = (state) => {
  const messages = state.messages
  const lastMessage = messages.at(-1)

  // If the LLM makes a tool call, then perform an action
  if (lastMessage?.tool_calls?.length) {
    return 'toolNode'
  }
  // Otherwise, we stop (reply to the user)
  return '__end__'
}

// Build workflow
const agentBuilder = new StateGraph(State)
  .addNode('llmCall', llmCall)
  .addNode('toolNode', toolNode)
  // Add edges to connect nodes
  .addEdge('__start__', 'llmCall')
  .addConditionalEdges('llmCall', shouldContinue, ['toolNode', '__end__'])
  .addEdge('toolNode', 'llmCall')
  .compile()

// Invode
const messages = [
  {
    role: 'user',
    content: 'Add 3 and 4 then multiply that by 20 and divide that by 2.'
  }
]
const result = await agentBuilder.invoke({ messages })
console.log('result', result.messages.at(-1).content)
