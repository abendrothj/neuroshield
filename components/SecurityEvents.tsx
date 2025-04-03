'use client';

import { useState, useEffect } from 'react';
import { api, SecurityEvent } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';

export function SecurityEvents() {
    const [events, setEvents] = useState<SecurityEvent[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [newEvent, setNewEvent] = useState<Omit<SecurityEvent, 'ipfshash'>>({
        id: '',
        timestamp: new Date().toISOString(),
        type: '',
        details: '',
    });

    useEffect(() => {
        loadEvents();
    }, []);

    const loadEvents = async () => {
        try {
            const data = await api.getAllEvents();
            setEvents(data);
        } catch (err) {
            setError('Failed to load events');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            const result = await api.logEvent(newEvent);
            setNewEvent({
                id: '',
                timestamp: new Date().toISOString(),
                type: '',
                details: '',
            });
            loadEvents();
        } catch (err) {
            setError('Failed to log event');
            console.error(err);
        }
    };

    if (loading) return <div>Loading...</div>;
    if (error) return <div className="text-red-500">{error}</div>;

    return (
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
                                        <p className="text-sm">{event.details}</p>
                                        {event.ipfshash && (
                                            <p className="text-xs text-gray-500">
                                                IPFS Hash: {event.ipfshash}
                                            </p>
                                        )}
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </CardContent>
            </Card>
        </div>
    );
} 