# Troubleshooting and Tips

## Overview

This section provides solutions to common issues you may encounter while working with AI-DLC and Kiro. Since AI-DLC is an evolving methodology, this guide focuses on general troubleshooting principles rather than rigid procedures.

------------------------------------------------------------------------

## Troubleshooting

### Problem: `the context window has overflowed`

**Solution:**

Start a new session of Kiro CLI and initiate a chat with the following prompt:

```
I am resuming a new session, please review the existing state of progress in the ai-dlc docs folder and continue with the workflow where I left off
```

**Recommendations**

If the team is using the Amazon Q Developer plugin in the IDE, we recommend using Kiro CLI instead as it has optimized context handling.

If the team is using Kiro CLI and experiences this error, you can view current context usage by typing `/usage` and then using the `/compact` command to summarize and compact the context.

## Tips

1.  **Ensure the team reviews all outputs produced by AI-DLC** - this will help ensure that the software being built adheres to the requirements and standards the team has and prevent incorrect assumptions from AI building software that is not what was expected.

2.  **Ask clarification questions when you don\'t understand something produced by AI-DLC** - if there is output that is not entirely clear, ask clarifying questions in the chat to help the team gain a better understanding of intent as well as improve the accuracy in the outputs. This will help ensure the software that is being built more closely aligns with the team\'s expectations.

3.  **Don\'t forget to commit all the AI-DLC artifacts to git for version control!** - Additionally, when working in a collaborative team environment, it is best practice to commit your local changes after each step so can rollback to a previous step if needed.

### Problem: When running the application, it doesn\'t appear in the preview window

> **Note:**
>
> If you are participating in an AWS guided event, your web IDE has been configured to redirect the web-proxy to the `/flappy-kiro` path.

Please configure your development environment with the following settings.

- Base path:
  /flappy-kiro
  .
- Port:
  8081
  .

This can be done by requesting this non-functional requirement into the Q chat window during the inception phase.

Suggested prompts

```
Configure the development environment to run with the following settings
* Base path: `flappy-kiro`* Port: `8081`
```

```
You are an Nginx server running behind a CloudFront proxy.
Examine the Nginx configuration under the `/etc/nginx/config.d` path, and configure the application so that port forwarding is used.
```

------------------------------------------------------------------------
