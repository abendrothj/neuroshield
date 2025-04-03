'use client';

import React, { useState, useEffect } from 'react';
import { SecurityEvent, ApiError } from '../types';
import { useWebSocket } from '../hooks/useWebSocket';
import { ErrorBoundary } from './ui/error-boundary';
import { LoadingSpinner } from './ui/loading-spinner';
import { Alert } from './ui/alert';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';

const SecurityEvents: React.FC = () => {
    const [events, setEvents] = useState<SecurityEvent[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<ApiError | null>(null);
    const { lastMessage, readyState } = useWebSocket();
    const [newEvent, setNewEvent] = useState<Omit<SecurityEvent, 'ipfshash'>>({
        id: '',
        timestamp: new Date().toISOString(),
        type: '',
        details: '',
    });

    useEffect(() => {
        const fetchEvents = async () => {
            try {
                setLoading(true);
                const response = await fetch('/api/events');
                if (!response.ok) {
                    const errorData: ApiError = await response.json();
                    throw new Error(errorData.message);
                }
                const data: SecurityEvent[] = await response.json();
                setEvents(data);
                setError(null);
            } catch (err) {
                setError({
                    code: 'FETCH_ERROR',
                    message: err instanceof Error ? err.message : 'Failed to fetch events'
                });
            } finally {
                setLoading(false);
            }
        };

        fetchEvents();
    }, []);

    useEffect(() => {
        if (lastMessage && lastMessage.type === 'event') {
            const newEvent = lastMessage.data as SecurityEvent;
            setEvents(prev => [newEvent, ...prev]);
        } else if (lastMessage && lastMessage.type === 'error') {
            const errorData = lastMessage.data as ApiError;
            setError(errorData);
        }
    }, [lastMessage]);

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'critical':
                return 'bg-red-100 text-red-800';
            case 'high':
                return 'bg-orange-100 text-orange-800';
            case 'medium':
                return 'bg-yellow-100 text-yellow-800';
            case 'low':
                return 'bg-green-100 text-green-800';
            default:
                return 'bg-gray-100 text-gray-800';
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            const result = await fetch('/api/events', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(newEvent),
            });
            if (!result.ok) {
                const errorData: ApiError = await result.json();
                throw new Error(errorData.message);
            }
            const data: SecurityEvent = await result.json();
            setNewEvent({
                id: '',
                timestamp: new Date().toISOString(),
                type: '',
                details: '',
            });
            setEvents(prev => [data, ...prev]);
        } catch (err) {
            setError({
                code: 'POST_ERROR',
                message: err instanceof Error ? err.message : 'Failed to log event'
            });
        }
    };

    if (loading) {
        return <LoadingSpinner />;
    }

    if (error) {
        return <Alert type="error" message={error.message} />;
    }

    return (
        <ErrorBoundary>
            <div className="space-y-6">
                <Card>
                    <CardHeader>
                        <CardTitle>Log New Security Event</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="id">Event ID</Label>
                                <Input
                                    id="id"
                                    value={newEvent.id}
                                    onChange={(e) => setNewEvent({ ...newEvent, id: e.target.value })}
                                    required
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="type">Event Type</Label>
                                <Input
                                    id="type"
                                    value={newEvent.type}
                                    onChange={(e) => setNewEvent({ ...newEvent, type: e.target.value })}
                                    required
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="details">Details</Label>
                                <Textarea
                                    id="details"
                                    value={newEvent.details}
                                    onChange={(e) => setNewEvent({ ...newEvent, details: e.target.value })}
                                    required
                                />
                            </div>
                            <Button type="submit">Log Event</Button>
                        </form>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle>Security Events</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            {events.map((event) => (
                                <Card key={event.id}>
                                    <CardContent className="pt-6">
                                        <div className="space-y-2">
                                            <div className="flex justify-between">
                                                <span className="font-semibold">{event.type}</span>
                                                <span className="text-sm text-gray-500">
                                                    {new Date(event.timestamp).toLocaleString()}
                                                </span>
                                            </div>
                                            <p className="text-sm">{event.details.description}</p>
                                            <span
                                                className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(
                                                    event.details.severity
                                                )}`}
                                            >
                                                {event.details.severity}
                                            </span>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            </div>
        </ErrorBoundary>
    );
};

export default SecurityEvents; 