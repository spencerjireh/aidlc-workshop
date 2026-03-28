# Examine the workshop files

## Overview

In this section you briefly review the project resources that are deployed for the workshop.

## Project rules

We will use [Amazon Q Developer project rules](https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/context-project-rules.html) to drive the AI-DLC process.

*Project Rules* are a way to build a library of coding standards and best practices that are automatically used as context when interacting with the assistant.

These rules are defined in Markdown files stored in your project's `.amazonq/rules` folder. Once created, they automatically become part of the context within your project, maintaining consistency across team members regardless of their experience level.

By storing these rules in your project, you can ensure consistency across developers, regardless of their experience level.

## (Optional) Examine the AI-DLC files

The AI-DLC files have been deployed in the `.amazonq` folder, and consist of markdown files which are used as context when interactive with Kiro on the command line.

Explore the markdown files in the `.amazonq` and sub folders to understand how these rules will ensure that your project follows the AI-DLC methodology.

## Examine the initial project

1.  Select the **Explorer** icon in the **View** panel to examine the files downloaded.

The `assets` folder is a game resources directory containing:

- `ghosty.png` - Character sprite image for the game.
- `game_over.wav` - Audio file for game over sound effect.
- `jump.wav` - Audio file for jump sound effect.

[![Initial Files](https://static.us-east-1.prod.workshops.aws/89665b58-5e6c-42f1-a3d4-67b9206a068a/static/images/examine-workshop/initial-files.png?Key-Pair-Id=K36Q2WVO3JP7QD&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9zdGF0aWMudXMtZWFzdC0xLnByb2Qud29ya3Nob3BzLmF3cy84OTY2NWI1OC01ZTZjLTQyZjEtYTNkNC02N2I5MjA2YTA2OGEvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3NTIxNTA5OX19fV19&Signature=cGtFPTwAScim9x7WgodUm6FQKqzDrC6640Ku7EP4pXgyk7TjL4u4PdLfB1WE3AHa%7ELH7p83cMiUCiHPoRtOH8OrboafUMxPoN6vTskLkQqryiKwc1kVUZUGxcQMTA46rIIQegkAPaS35uvLcvey3G6RCf3Mn6AGx97y1kAhaefpXxeaXa6TPZfeIRd7tGWDQIHiH4y6a0apXGPoT6bPkdhm-C5v58J1li5XI7GFfJXqWROFe-wYaeyGKOpcoqIAjybRBBlXBRBmggVYtCijpbxaMi4lCFQMQBc8Qyv5R1h43j7eDnhDrRtXpqNK54SzI2HRyN%7EHbrw48dD4aHZv6OA__)]

The `img` folder contains `example-ui.png`, an example of the Flappy Kiro game interface.

## Summary

You have examined the files provisioned for the workshop. Next you will create an application using the AI-DLC workflow.
