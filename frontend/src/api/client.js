import axios from 'axios';

const client = axios.create({
  baseURL: '/api/v1',
});

// Segments
export function getSegments() {
  return client.get('/segments/').then((res) => res.data);
}

export function getSegment(segmentId) {
  return client.get(`/segments/${segmentId}`).then((res) => res.data);
}

export function createSegments(params) {
  return client.post('/segments/create', params).then((res) => res.data);
}

export function getSegmentDistribution() {
  return client.get('/analytics/segments/distribution').then((res) => res.data);
}

// Campaigns
export function getCampaigns() {
  return client.get('/campaigns/').then((res) => res.data);
}

export function getCampaign(campaignId) {
  return client.get(`/campaigns/${campaignId}`).then((res) => res.data);
}

export function createCampaign(payload) {
  return client.post('/campaigns/create', payload).then((res) => res.data);
}

export function activateCampaign(campaignId) {
  return client.post(`/campaigns/${campaignId}/activate`).then((res) => res.data);
}

export function getCampaignReach(campaignId) {
  return client.get(`/campaigns/${campaignId}/reach`).then((res) => res.data);
}

export function getCampaignPerformance(campaignId) {
  return client.get(`/campaigns/${campaignId}/performance`).then((res) => res.data);
}

// Ads
export function generateAds(payload) {
  return client.post('/ads/generate', payload).then((res) => res.data);
}

// Chatbot
export function sendChatQuery(payload) {
  return client.post('/chatbot/query', payload).then((res) => res.data);
}

export function getConversationContext(sessionId) {
  return client.get(`/chatbot/context/${sessionId}`).then((res) => res.data);
}

// Data ingestion
export function ingestCustomers(payload) {
  return client.post('/data/ingest', payload).then((res) => res.data);
}

// Config
export function configLLM(payload) {
  return client.post('/config/llm', payload).then((res) => res.data);
}

export function listProviders() {
  return client.get('/config/providers').then((res) => res.data);
}

export default client;
