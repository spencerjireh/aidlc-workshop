"""LLM Engine for orchestrating LLM operations."""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.models.llm import LLMConfiguration, LLMParameters, LLMProvider
from src.models.ml import ClusterStatistics
from src.models.segment import SegmentProfile
from src.models.campaign import AdContent, AdFormat
from src.models.chatbot import QueryIntent, ChatbotResponse, ConversationContext
from src.models.customer import CustomerProfile
from src.engines.adapters import LLMProviderAdapter, OpenAIAdapter, AnthropicAdapter, LocalModelAdapter

logger = logging.getLogger(__name__)


class LLMEngine:
    """Engine for orchestrating LLM operations with retry logic and provider abstraction."""
    
    def __init__(self, configuration: LLMConfiguration):
        """Initialize LLM Engine with configuration.
        
        Args:
            configuration: LLM configuration including provider, model, and parameters.
        """
        self.configuration = configuration
        self.adapter = self._create_adapter()
        self.request_log: List[Dict[str, Any]] = []
    
    def _create_adapter(self) -> LLMProviderAdapter:
        """Create appropriate adapter based on provider configuration.
        
        Returns:
            LLMProviderAdapter: Initialized provider adapter.
            
        Raises:
            ValueError: If provider is not supported.
        """
        provider = self.configuration.provider
        
        if provider == LLMProvider.OPENAI:
            return OpenAIAdapter(
                api_key=self.configuration.api_key,
                model_name=self.configuration.model_name
            )
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicAdapter(
                api_key=self.configuration.api_key,
                model_name=self.configuration.model_name
            )
        elif provider == LLMProvider.LOCAL:
            if not self.configuration.api_endpoint:
                raise ValueError("api_endpoint is required for LOCAL provider")
            return LocalModelAdapter(
                endpoint=self.configuration.api_endpoint,
                model_name=self.configuration.model_name
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def call_llm(
        self,
        prompt: str,
        parameters: Optional[LLMParameters] = None,
        max_retries: int = 3
    ) -> str:
        """Call LLM with retry logic and exponential backoff.
        
        Args:
            prompt: Input prompt for the LLM.
            parameters: Generation parameters (uses config defaults if None).
            max_retries: Maximum number of retry attempts (default: 3).
            
        Returns:
            str: Generated text response.
            
        Raises:
            Exception: If all retry attempts fail.
        """
        if parameters is None:
            parameters = self.configuration.parameters
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Call the adapter
                response = self.adapter.generate(prompt, parameters)
                
                elapsed_time = time.time() - start_time
                
                # Log request and response
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "provider": self.adapter.get_provider_name(),
                    "model": self.configuration.model_name,
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                    "parameters": parameters.model_dump(),
                    "elapsed_time_seconds": elapsed_time,
                    "attempt": attempt + 1,
                    "success": True
                }
                self.request_log.append(log_entry)
                
                logger.info(f"LLM call successful on attempt {attempt + 1}")
                return response
                
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM call failed on attempt {attempt + 1}/{max_retries}: {e}")
                
                # Log failed attempt
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "provider": self.adapter.get_provider_name(),
                    "model": self.configuration.model_name,
                    "prompt_length": len(prompt),
                    "parameters": parameters.model_dump(),
                    "attempt": attempt + 1,
                    "success": False,
                    "error": str(e)
                }
                self.request_log.append(log_entry)
                
                # Exponential backoff: wait 2^attempt seconds
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All retries failed
        logger.error(f"All {max_retries} LLM call attempts failed")
        raise last_exception or Exception("LLM call failed after all retries")
    
    def generate_segment_profile(
        self,
        cluster_statistics: ClusterStatistics,
        cluster_id: int
    ) -> SegmentProfile:
        """Generate descriptive name and profile for a customer segment.
        
        Args:
            cluster_statistics: Statistics for the cluster.
            cluster_id: Cluster identifier.
            
        Returns:
            SegmentProfile: Generated segment profile with name and description.
        """
        # Build prompt from cluster statistics
        prompt = f"""Based on the following customer cluster statistics, generate a descriptive name and detailed profile for this customer segment.

Cluster Statistics:
- Cluster ID: {cluster_statistics.cluster_id}
- Size: {cluster_statistics.size} customers
- Average Age: {cluster_statistics.average_age:.1f} years
- Average Transaction Frequency: {cluster_statistics.average_transaction_frequency:.1f} transactions/month
- Average Transaction Value: ${cluster_statistics.average_transaction_value:.2f}
- Top Merchant Categories: {', '.join([f"{cat} ({count})" for cat, count in cluster_statistics.top_merchant_categories[:5]])}
- Location Distribution: {', '.join([f"{loc} ({count})" for loc, count in list(cluster_statistics.location_distribution.items())[:5]])}
- Preferred Payment Methods: {', '.join([f"{method} ({count})" for method, count in list(cluster_statistics.preferred_payment_methods.items())[:3]])}

Please provide:
1. A short, memorable name for this segment (2-4 words)
2. A detailed description (2-3 sentences) explaining the segment's characteristics
3. Key differentiating factors that make this segment unique

Format your response as:
NAME: [segment name]
DESCRIPTION: [detailed description]
DIFFERENTIATING_FACTORS: [factor 1], [factor 2], [factor 3]
"""
        
        response = self.call_llm(prompt)
        
        # Parse response
        name = "Unnamed Segment"
        description = "No description available"
        differentiating_factors = []
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('NAME:'):
                name = line.replace('NAME:', '').strip()
            elif line.startswith('DESCRIPTION:'):
                description = line.replace('DESCRIPTION:', '').strip()
            elif line.startswith('DIFFERENTIATING_FACTORS:'):
                factors_str = line.replace('DIFFERENTIATING_FACTORS:', '').strip()
                differentiating_factors = [f.strip() for f in factors_str.split(',')]
        
        return SegmentProfile(
            segment_id=f"segment_{cluster_id}",
            name=name,
            description=description,
            differentiating_factors=differentiating_factors
        )

    def explain_cluster_assignment(
        self,
        customer: CustomerProfile,
        cluster_centroid: List[float],
        top_features: List[tuple[str, float]]
    ) -> str:
        """Generate natural language explanation for customer-segment assignment.
        
        Args:
            customer: Customer profile.
            cluster_centroid: Cluster centroid in PCA space.
            top_features: List of (feature_name, importance) tuples.
            
        Returns:
            str: Natural language explanation.
        """
        # Build prompt with customer data and top features
        features_str = '\n'.join([f"- {name}: importance {importance:.3f}" for name, importance in top_features[:3]])
        
        prompt = f"""Explain why this customer was assigned to their segment based on the following information:

Customer Profile:
- Age: {customer.age}
- Location: {customer.location}
- Transaction Frequency: {customer.transaction_frequency} transactions/month
- Average Transaction Value: ${customer.average_transaction_value:.2f}
- Total Spend: ${customer.total_spend:.2f}
- Top Merchant Categories: {', '.join(customer.merchant_categories[:3])}

Top Contributing Features:
{features_str}

Provide a clear, concise explanation (2-3 sentences) of why this customer belongs to this segment, focusing on the most important factors.
"""
        
        return self.call_llm(prompt)
    
    def generate_ad_content(
        self,
        segment_profile: SegmentProfile,
        campaign_brief: str,
        format: AdFormat,
        num_variations: int = 3
    ) -> List[AdContent]:
        """Generate personalized ad content for a segment.
        
        Args:
            segment_profile: Profile of the target segment.
            campaign_brief: Campaign description and objectives.
            format: Ad format (SHORT, MEDIUM, LONG).
            num_variations: Number of ad variations to generate (default: 3).
            
        Returns:
            List[AdContent]: Generated ad content variations.
        """
        # Determine character limit based on format
        format_limits = {
            AdFormat.SHORT: 50,
            AdFormat.MEDIUM: 150,
            AdFormat.LONG: 300
        }
        char_limit = format_limits[format]
        
        prompt = f"""Generate {num_variations} variations of ad copy for an e-wallet platform targeting the following customer segment:

Segment Profile:
- Name: {segment_profile.name}
- Description: {segment_profile.description}
- Key Characteristics: {', '.join(segment_profile.differentiating_factors)}

Campaign Brief:
{campaign_brief}

Requirements:
- Each ad must be {char_limit} characters or less
- Focus on e-wallet use cases (cashback, merchant promotions, payment convenience)
- Tailor messaging to the segment's characteristics
- Create {num_variations} distinct variations for A/B testing

Format your response as:
AD1: [ad copy 1]
AD2: [ad copy 2]
AD3: [ad copy 3]
"""
        
        response = self.call_llm(prompt)
        
        # Parse response
        ad_contents = []
        for i in range(1, num_variations + 1):
            marker = f"AD{i}:"
            if marker in response:
                # Extract ad text
                start_idx = response.index(marker) + len(marker)
                # Find next marker or end of string
                next_marker = f"AD{i+1}:"
                if next_marker in response:
                    end_idx = response.index(next_marker)
                else:
                    end_idx = len(response)
                
                ad_text = response[start_idx:end_idx].strip()
                
                # Truncate if exceeds limit
                if len(ad_text) > char_limit:
                    ad_text = ad_text[:char_limit]
                
                # Determine use case from campaign brief
                use_case = "payment_convenience"
                if "cashback" in campaign_brief.lower():
                    use_case = "cashback"
                elif "merchant" in campaign_brief.lower() or "promo" in campaign_brief.lower():
                    use_case = "merchant_promo"
                
                ad_content = AdContent(
                    ad_id=f"ad_{segment_profile.segment_id}_{format.value}_{i}",
                    segment_id=segment_profile.segment_id,
                    campaign_id="",  # To be set by caller
                    format=format,
                    content=ad_text,
                    variation_number=i,
                    use_case=use_case
                )
                ad_contents.append(ad_content)
        
        return ad_contents
    
    def interpret_query(
        self,
        query: str,
        context: ConversationContext
    ) -> QueryIntent:
        """Interpret natural language query from chatbot.
        
        Args:
            query: User's natural language query.
            context: Conversation context for understanding.
            
        Returns:
            QueryIntent: Structured query intent.
        """
        # Build prompt with conversation history
        history_str = ""
        if context.conversation_history:
            recent_messages = context.conversation_history[-3:]  # Last 3 messages
            history_str = "\n".join([f"{msg.role.value}: {msg.content}" for msg in recent_messages])
        
        prompt = f"""Analyze the following user query and extract the intent and entities.

Conversation History:
{history_str if history_str else "No previous conversation"}

Current Query: {query}

Determine:
1. Query Type: segment_info, comparison, performance, trend, customer_count, or top_categories
2. Entities: Extract segment names, dates, metrics, or other relevant entities
3. Confidence: How confident are you in this interpretation (0.0 to 1.0)

Format your response as:
TYPE: [query_type]
ENTITIES: [entity1=value1, entity2=value2, ...]
CONFIDENCE: [0.0-1.0]
"""
        
        response = self.call_llm(prompt)
        
        # Parse response
        query_type = "segment_info"
        entities = {}
        confidence = 0.8
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('TYPE:'):
                query_type = line.replace('TYPE:', '').strip()
            elif line.startswith('ENTITIES:'):
                entities_str = line.replace('ENTITIES:', '').strip()
                if entities_str and entities_str != "None":
                    for entity in entities_str.split(','):
                        if '=' in entity:
                            key, value = entity.split('=', 1)
                            entities[key.strip()] = value.strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                except ValueError:
                    confidence = 0.8
        
        from src.models.chatbot import QueryType
        # Map string to QueryType enum
        query_type_map = {
            "segment_info": QueryType.SEGMENT_INFO,
            "comparison": QueryType.COMPARISON,
            "performance": QueryType.PERFORMANCE,
            "trend": QueryType.TREND,
            "customer_count": QueryType.CUSTOMER_COUNT,
            "top_categories": QueryType.TOP_CATEGORIES
        }
        
        intent_type = query_type_map.get(query_type, QueryType.SEGMENT_INFO)
        
        return QueryIntent(
            intent_type=intent_type,
            entities=entities,
            confidence=confidence
        )
    
    def generate_response(
        self,
        query_intent: QueryIntent,
        data: Dict[str, Any],
        context: ConversationContext
    ) -> ChatbotResponse:
        """Generate natural language response for chatbot.
        
        Args:
            query_intent: Structured query intent.
            data: Retrieved data to include in response.
            context: Conversation context.
            
        Returns:
            ChatbotResponse: Generated response with text and optional visualizations.
        """
        start_time = time.time()
        
        # Build prompt with query intent and data
        data_str = str(data)[:1000]  # Limit data size in prompt
        
        prompt = f"""Generate a natural language response to the user's query based on the following information:

Query Intent: {query_intent.intent_type.value}
Entities: {query_intent.entities}

Retrieved Data:
{data_str}

Provide:
1. A clear, conversational response (2-4 sentences)
2. Suggested follow-up questions (2-3 questions)
3. Visualization type if applicable (table, chart, bar, pie, line, or none)

Format your response as:
RESPONSE: [natural language response]
FOLLOWUPS: [question 1], [question 2], [question 3]
VISUALIZATION: [visualization_type or none]
"""
        
        response = self.call_llm(prompt)
        
        # Parse response
        response_text = "I couldn't generate a response."
        followups = []
        visualization_type = None
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('RESPONSE:'):
                response_text = line.replace('RESPONSE:', '').strip()
            elif line.startswith('FOLLOWUPS:'):
                followups_str = line.replace('FOLLOWUPS:', '').strip()
                followups = [q.strip() for q in followups_str.split(',')]
            elif line.startswith('VISUALIZATION:'):
                viz = line.replace('VISUALIZATION:', '').strip().lower()
                if viz != "none":
                    visualization_type = viz
        
        elapsed_time = int((time.time() - start_time) * 1000)
        
        return ChatbotResponse(
            response_id=f"response_{int(time.time())}",
            session_id=context.session_id,
            text=response_text,
            data=data,
            visualization_type=visualization_type,
            suggested_followups=followups,
            response_time_ms=elapsed_time
        )
    
    def get_request_log(self) -> List[Dict[str, Any]]:
        """Get log of all LLM requests and responses.
        
        Returns:
            List[Dict]: Log entries with timestamps, prompts, responses, and metadata.
        """
        return self.request_log
