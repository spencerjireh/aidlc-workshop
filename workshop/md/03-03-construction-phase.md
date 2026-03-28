# Construction Phase

The Construction phase transforms your validated requirements into working code through AI-driven development and testing. You\'ll achieve a fully functional application ready for deployment.

The Construction phase focuses on the Mob Programming and Mob Testing rituals:

- AI models the core business logic for the assigned Unit using [Domain-Driven Design](https://docs.aws.amazon.com/prescriptive-guidance/latest/hexagonal-architectures/overview.html#ddd) principles. Example: For the \"Recommendation Algorithm\" Unit, AI identifies relevant entities like `Product`, `Customer`, and `Purchase History `and their relationships.

- The team reviews and validates the domain models, refining business logic and ensuring alignment with real-world scenarios (e.g., how to handle missing purchase history for new customers)

- AI translates the domain models into logical designs, applying Non-Functional Requirements (NFRs) like scalability and fault tolerance. Example: AI recommends architectural patterns (e.g., event-driven design) and technologies (e.g., AWS Lambda for serverless computation).

- The team evaluates AI's recommendations, approves trade-offs, and suggests additional considerations if needed. An example may be that you **accept** a recommendation to use Lambda for scalability, but you **override** the suggestion to use DynamoDB for fast query performance.

- AI generates executable code for each Unit, mapping logical components to specific AWS services.

- AI auto-generates functional, security, and performance tests (e.g., For the \"Recommendation Algorithm\" Unit, AI creates code to implement collaborative filtering and integrates it with a DynamoDB data source)

- The team reviews the generated code and test scenarios/cases, making adjustments where necessary to ensure quality and compliance.

Once the Construction phase is complete, you\'ll move onto the Operations phase.
