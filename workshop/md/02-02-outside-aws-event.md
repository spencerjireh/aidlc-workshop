# Outside an AWS event

> **Note:**
>
> This section only applies if you are running the workshop in your own account. Click [here](/event/dashboard/en-US/workshop/10-start-workshop/11-aws-event) if you are participating in an AWS guided event.

> **Note:**
>
> Make sure to follow the [Clean Up Resources](/event/dashboard/en-US/workshop/90-cleanup) section to remove workshop-related resources at the end to avoid unnecessary charges.

## Prerequisites

For this workshop, you will need an Integrated Development Environment (IDE) installed on your local machine. It is recommended you use [Visual Studio Code](https://code.visualstudio.com/download).

## IDE configuration

The following items are required in your IDE:

- [Kiro on the command line](https://kiro.dev/cli/)

<!-- -->

- [Git](https://github.com/git-guides/).

To Install Kiro on the command line, refer to the [Kiro CLI documentation](https://kiro.dev/cli/).

You will need to have installed any programming languages or other dependencies needed for your application.

> **Note:**
>
> If Kiro on the command line detects missing software, it will ask for your approval before installing the software and proceeding.

## Create a blank project

1.  In your IDE, open a terminal window and navigate to the a directory under which you will create the workshop.

2.  Run the following command to create a `aidlc-workshop` directory.

```
mkdir aidlc-workshop
```

3.  Navigate to the project folder in your IDE by selecting **File** \> **Open Folder\...**, and selecting the `aidc-workshop` folder.

## Download and unzip the workshop materials

Run the following command to download and extract the files used in the workshop.

```
curl'https://static.us-east-1.prod.workshops.aws/89665b58-5e6c-42f1-a3d4-67b9206a068a/assets/qdev-aidlc.zip?Key-Pair-Id=K36Q2WVO3JP7QD&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9zdGF0aWMudXMtZWFzdC0xLnByb2Qud29ya3Nob3BzLmF3cy84OTY2NWI1OC01ZTZjLTQyZjEtYTNkNC02N2I5MjA2YTA2OGEvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3NTIxNTA5OX19fV19&Signature=cGtFPTwAScim9x7WgodUm6FQKqzDrC6640Ku7EP4pXgyk7TjL4u4PdLfB1WE3AHa~LH7p83cMiUCiHPoRtOH8OrboafUMxPoN6vTskLkQqryiKwc1kVUZUGxcQMTA46rIIQegkAPaS35uvLcvey3G6RCf3Mn6AGx97y1kAhaefpXxeaXa6TPZfeIRd7tGWDQIHiH4y6a0apXGPoT6bPkdhm-C5v58J1li5XI7GFfJXqWROFe-wYaeyGKOpcoqIAjybRBBlXBRBmggVYtCijpbxaMi4lCFQMQBc8Qyv5R1h43j7eDnhDrRtXpqNK54SzI2HRyN~Hbrw48dD4aHZv6OA__' -o qdev-aidlc.zip
unzip -o qdev-aidlc.zip -d .&&rm qdev-aidlc.zip
```

## Initialize a local Git repo

1.  Initialize the folder with an empty local git repository.

```
git init
```

2.  Make an initial commit to your local git repository:

```
gitadd.git commit -m 'Initial commit'
```

## Summary

You have successfully configured your IDE, including setting up a project directory, downloading the workshop files, and initialized a local git repository.

Next, you will connect to Kiro.
