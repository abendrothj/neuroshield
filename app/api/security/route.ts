import { NextResponse } from 'next/server';
import { FabricService } from '@/backend/services/fabric.service';
import { AIService } from '@/backend/services/ai.service';

const fabricService = new FabricService();
const aiService = new AIService();

// Initialize services
async function initializeServices() {
    try {
        await fabricService.initialize();
        await aiService.initialize();
    } catch (error) {
        console.error('Failed to initialize services:', error);
    }
}

// Initialize on module load
initializeServices();

export async function POST(request: Request) {
    try {
        const data = await request.json();
        
        // Detect threat using AI
        const threatResult = await aiService.detectThreat(data);
        
        if (threatResult.isThreat) {
            // Log security event on blockchain
            const event = await fabricService.logSecurityEvent(
                'THREAT_DETECTED',
                threatResult.severity,
                threatResult.description,
                '', // IPFS hash will be added later
                {
                    confidence: threatResult.confidence,
                    ...threatResult.metadata
                }
            );

            return NextResponse.json({
                success: true,
                event,
                threat: threatResult
            });
        }

        return NextResponse.json({
            success: true,
            threat: threatResult
        });
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

        if (eventId) {
            // Query specific event
            const event = await fabricService.queryEvent(eventId);
            return NextResponse.json({ success: true, event });
        } else {
            // Query all events
            const events = await fabricService.queryAllEvents();
            return NextResponse.json({ success: true, events });
        }
    } catch (error) {
        console.error('Error querying security events:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to query security events' },
            { status: 500 }
        );
    }
} 