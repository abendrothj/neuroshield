const request = require('supertest')
const app = require('../src/server')

describe('Server', () => {
  describe('GET /health', () => {
    it('responds with 200 and healthy status', async () => {
      const response = await request(app).get('/health')
      expect(response.statusCode).toBe(200)
      expect(response.body).toEqual({ status: 'healthy' })
    })
  })

  describe('POST /api/events', () => {
    it('logs an event successfully', async () => {
      const eventData = {
        type: 'THREAT_DETECTED',
        severity: 'HIGH',
        details: {
          source: '192.168.1.100',
          timestamp: new Date().toISOString(),
        },
      }

      const response = await request(app)
        .post('/api/events')
        .send(eventData)
        .set('Accept', 'application/json')

      expect(response.statusCode).toBe(201)
      expect(response.body).toHaveProperty('id')
      expect(response.body.type).toBe(eventData.type)
    })

    it('validates required fields', async () => {
      const response = await request(app)
        .post('/api/events')
        .send({})
        .set('Accept', 'application/json')

      expect(response.statusCode).toBe(400)
      expect(response.body).toHaveProperty('error')
    })
  })
}) 