import { NextResponse } from 'next/server';

// Get the backend URL from environment variables
const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

export async function POST(request: Request) {
    try {
        const data = await request.json();
        
        // Forward the request to the backend
        const response = await fetch(`${BACKEND_URL}/api/security/threat-detection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        
        if (!response.ok) {
            throw new Error(`Backend responded with ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        return NextResponse.json(result);
    } catch (error) {
        console.error('Error processing security event:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to process security event' },
            { status: 500 }
        );
    }
}

export async function GET(request: Request) {
    try {
        const { searchParams } = new URL(request.url);
        const eventId = searchParams.get('eventId');
        
        let endpoint = `${BACKEND_URL}/api/security/events`;
        if (eventId) {
            endpoint += `/${eventId}`;
        }
        
        const response = await fetch(endpoint);
        
        if (!response.ok) {
            throw new Error(`Backend responded with ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        return NextResponse.json(result);
    } catch (error) {
        console.error('Error querying security events:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to query security events' },
            { status: 500 }
        );
    }
} 