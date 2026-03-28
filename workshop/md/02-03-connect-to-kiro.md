# Connect to Kiro

## Overview

In this section, you will connect to Kiro using an AWS Builder ID, which will allow access to certain developer tools and services from AWS without requiring an AWS account. AWS Builder IDs are free to use.

You can access details about your AWS Builder ID in [AWS Builder ID profile page](https://profile.aws.amazon.com/).

## Authenticate to Kiro on the Command Line

For this workshop we will use [Kiro on the command line](https://kiro.dev/docs/cli/)

> **Note:**
>
> If you are running in your own account, ensure you have Kiro on the command line installed.

1.  Run the following command in the IDE terminal:

```
kiro-cli login --use-device-flow
```

2.  For **Select login method**, choose **Use for Free with Builder ID**.
3.  Holding the **Command** key (Mac) or **Control** key (windows), select the URL. Chose **Open** to open a new browser tab.
4.  For **Authorization requested**, choose **Confirm and continue**.

> **Note:**
>
> If you did not use the Builder ID to connect to the workshop, or do not have one already setup, follow the prompts to setup a new Builder ID.

5.  Choose **Allow access**.

6.  The window will show **Request approved**. Close the tab and return back to your terminal.

7.  Test you are connected using the following command:

```
kiro-cli whoami
```

(Optional) Connect to Amazon Q chat using an AWS Builder ID

This workshop does not use the [Amazon Q Developer extension for the IDE](https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/q-in-IDE.html), however you can also enable it to provide additional Amazon Q Developer features. Refer to the Amazon Q documentation for more details.

1.  In the navigation bar, select the **Amazon Q** icon.

2.  For **Choose a sign-in option**, select **Personal account**. Choose **Continue**.

[![Amazon Q extension IDE set up](https://static.us-east-1.prod.workshops.aws/89665b58-5e6c-42f1-a3d4-67b9206a068a/static/images/builder-id/ide-sign-in.png?Key-Pair-Id=K36Q2WVO3JP7QD&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9zdGF0aWMudXMtZWFzdC0xLnByb2Qud29ya3Nob3BzLmF3cy84OTY2NWI1OC01ZTZjLTQyZjEtYTNkNC02N2I5MjA2YTA2OGEvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3NTIxNTA5OX19fV19&Signature=cGtFPTwAScim9x7WgodUm6FQKqzDrC6640Ku7EP4pXgyk7TjL4u4PdLfB1WE3AHa%7ELH7p83cMiUCiHPoRtOH8OrboafUMxPoN6vTskLkQqryiKwc1kVUZUGxcQMTA46rIIQegkAPaS35uvLcvey3G6RCf3Mn6AGx97y1kAhaefpXxeaXa6TPZfeIRd7tGWDQIHiH4y6a0apXGPoT6bPkdhm-C5v58J1li5XI7GFfJXqWROFe-wYaeyGKOpcoqIAjybRBBlXBRBmggVYtCijpbxaMi4lCFQMQBc8Qyv5R1h43j7eDnhDrRtXpqNK54SzI2HRyN%7EHbrw48dD4aHZv6OA__)]

3.  At the prompt **Confirm Code for AWS Builder ID**, choose **Proceed to Browser**.
4.  At the prompt **Do you want code-server to open the external website?**, choose **Open**.
5.  A browser tab will open displaying the **Authorization requested** page. The confirmation code should already be populated. Choose **Confirm and continue**.

> **Note:**
>
> If you did not use the Builder ID to connect to the workshop, or do not have one already setup, follow the prompts to setup a new Builder ID.

6.  A browser tab will open displaying the **Allow AWS IDE Extensions for VSCode to access your data?** page. Choose **Allow access**.
7.  Close the tab and return to code-server tab.

------------------------------------------------------------------------

## Summary

You have configured your IDE and connected to Kiro on the command line.

Next you will briefly examine the files used in the workshop, before moving to the inception, construction and operation phases!
