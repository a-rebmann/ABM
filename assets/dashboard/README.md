# Agent Data and Dashboard Import

This guide walks you through the steps for importing the agent event log data and the preconfigured dashboard into SAP Signavio Process Intelligence.

1. Create a new process in **SAP Signavio Process Intelligence**.
2. In **Process Settings > Data**, upload the CSV file `Case Study-Agent Event Log.csv` under **Step 1: Import data**.
3. A new **Import Data** view will open, prompting you to define the data types for each column. Proceed as follows: 
   1. Select `case_id` as the **Case ID** column (you may need to scroll the table horizontally). Click **Next** at the top.
   2. Select `concept:instance` as the **Activity name** column and click **Next** at the top.
   3. Select `time:timestamp` as the **End timestamp** column and click **Done** at the top.
   4. Configure the following columns via their respective **Type** dropdown: 
      1. `ai:input_message` → **Text**
      2. `ai:message` → **Text**
      3. `ai:response_message` → **Text**
      4. `ai:response_thought` → **Text**
      5. `ai:tool_arguments` → **Text**
      6. `ai:tool_response` → **Text**
      7. `identity:id` → **Text**
   5. Click **Accept all suggestions** (top right) to auto-configure the remaining columns.
   6. Click **Import** (bottom right) to proceed.
4. After the data has been imported successfully, return to the process and open the **Dashboards** tab. Click **Import** and select the file `Case Study-Analysis Dashboard.json`.