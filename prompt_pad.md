# Prompt Pad

This file contains a collection of quick and effecient prompt to help you navigate the codebase and implement new features. Prompts are grouped by topic and are intended to be used as a starting point for new conversations or task generations.

New prompts may be added to this file as needed, but please follow the same format and style as the existing prompts utilizing the CodeBase System Documents.

## Prompting Upkeep

### Prompting a New Conversation to Utilize the Docs

Review the @.windsurfrules file, adhere to this strict workflow to ensure code is implemented gracefully. Remember that you have comprehensive documentation in @System_Integration.md @System_Architecture.md @System_Directory_Tree.md you may review if needed for applicable integration with the codebase. Finally, review @System_Scratchpad.md and continue on the next step on the Tasklist

## Implementing Missing Functionality

1. in @converter_interface.py  we need to use these imports, not remove them. Implement the missing functionality to properly utilize these imported types. Review @ASCII_UI_Implementation_Plan.md , Rewview @ascii_ui.py for context

2. Enhance the code to properly use the imports, particularly focusing on ConverterTier, ResourceType, EfficiencyFactor, and OptimizationSuggestion:
  - Look at what's missing in the current implementation
  - Implement proper usage of ConverterTier, ResourceType, EfficiencyFactor, and OptimizationSuggestion
  - Ensure all imported types are actually being used somewhere in the code. Ensure to implement these with context of the actual code already implemented.

## ASCII_UI_Implementation_Plan.md

- The converter_interface.py file imports several types from converter_models.py, but some of them aren't being properly utilized.
- Properly use ConverterTier, ResourceType, EfficiencyFactor, and OptimizationSuggestion.

### Implement the missing functionality:

See the specific implementation of the update_info_box and other methods in the ConverterDetailsView class to properly implement the missing functionality. Let me view those sections:

See that the ConverterDetailsView class already has some implementation for displaying the converter tier, but it's not fully utilizing the ConverterTier enum. Let me enhance the code to properly use all the imported types (ConverterTier, ResourceType, EfficiencyFactor, and OptimizationSuggestion).