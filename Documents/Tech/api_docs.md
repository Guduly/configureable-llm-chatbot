# API Documentation

## Authentication

### OAuth 2.0
All API requests require OAuth 2.0 authentication.

**Endpoint:** `https://api.example.com/oauth/token`

**Request:**
```json
{
  "client_id": "your_client_id",
  "client_secret": "your_secret",
  "grant_type": "client_credentials"
}
```

**Response:**
```json
{
  "access_token": "eyJhbG...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

## Rate Limits

- **Free Tier:** 100 requests per minute
- **Pro Tier:** 1000 requests per minute
- **Enterprise:** Unlimited

Rate limit headers:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Time when limit resets (Unix timestamp)

## Endpoints

### GET /api/v1/users
Retrieve user information.

**Parameters:**
- `id` (required): User ID
- `fields` (optional): Comma-separated list of fields to return

**Example:**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://api.example.com/api/v1/users?id=12345
```

**Response:**
```json
{
  "id": 12345,
  "name": "John Doe",
  "email": "john@example.com",
  "created_at": "2024-01-15T10:30:00Z"
}
```