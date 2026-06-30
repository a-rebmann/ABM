# Procurement Agent System — ABM Event Log & Analysis Dashboard

This folder contains the artifact data accompanying the demo paper:

> **"What Are Your Agents Up To? Agent Behavior Mining in Action"**  
> Hoang Vu, Maximilian Körner, Adrian Rebmann, Gabriel Kevorkian, Michael Perscheid, Gregor Berg, Timotheus Kampik  
> Submitted to *BPM 2026 Demos and Resources Forum*

## Contents

| File | Description |
|------|-------------|
| `abm_procurement-agent-system_200-traces.csv` | ABM-conformant event log for a simulated procurement agent system, containing 200 cases. Each row is an event capturing an action in the agent system, such as LLM calls or tool invocations, with attributes following the [Agent Behavior Mining (ABM) data model](https://arxiv.org/abs/2606.20669) in XES-compatible CSV format. |
| `Agent-Analysis-Dashboard_Procurement-Agent-System.json` | Export of the SAP Signavio Process Intelligence analysis dashboard used in the demo. The dashboard applies process discovery, conformance checking, and performance analysis directly to the ABM event log to provide operational visibility into agent behavior. |

## Context

The demo paper shows how GenAI agent behavior can be monitored and analyzed using established process mining techniques.
The procurement agent system simulates a multi-agent workflow for processing purchase requests, and the event log captures the agents' internal steps across 200 simulated cases.
The Signavio dashboard export can be imported into SAP Signavio Process Intelligence to reproduce the analysis shown in the paper.
