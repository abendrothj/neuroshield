import { useState, useEffect, useRef, useCallback } from 'react';
import { WebSocketMessage } from '../types';

export const useWebSocket = () => {
    const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
    const [readyState, setReadyState] = useState<number>(WebSocket.CONNECTING);
    const ws = useRef<WebSocket | null>(null);

    const connect = useCallback(() => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        ws.current = new WebSocket(wsUrl);

        ws.current.onopen = () => {
            setReadyState(WebSocket.OPEN);
        };

        ws.current.onclose = () => {
            setReadyState(WebSocket.CLOSED);
            // Attempt to reconnect after 5 seconds
            setTimeout(connect, 5000);
        };

        ws.current.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        ws.current.onmessage = (event) => {
            try {
                const message: WebSocketMessage = JSON.parse(event.data);
                setLastMessage(message);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };
    }, []);

    useEffect(() => {
        connect();

        return () => {
            if (ws.current) {
                ws.current.close();
            }
        };
    }, [connect]);

    const sendMessage = useCallback((message: WebSocketMessage) => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify(message));
        }
    }, []);

    return {
        lastMessage,
        readyState,
        sendMessage
    };
}; 