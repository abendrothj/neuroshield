export interface SecurityEvent {
    id: string;
    timestamp: string;
    type: string;
    details: {
        source: string;
        destination: string;
        protocol: string;
        severity: 'low' | 'medium' | 'high' | 'critical';
        description: string;
    };
    ipfsHash: string;
}

export interface ThreatStats {
    totalThreats: number;
    threatsByType: {
        [key: string]: number;
    };
    threatsBySeverity: {
        low: number;
        medium: number;
        high: number;
        critical: number;
    };
}

export interface SystemHealth {
    status: 'healthy' | 'degraded' | 'unhealthy';
    components: {
        blockchain: {
            status: 'healthy' | 'degraded' | 'unhealthy';
            lastBlock: number;
            peers: number;
        };
        ai: {
            status: 'healthy' | 'degraded' | 'unhealthy';
            modelVersion: string;
            accuracy: number;
        };
        storage: {
            status: 'healthy' | 'degraded' | 'unhealthy';
            ipfsNodes: number;
            usedSpace: number;
        };
    };
}

export interface UserSettings {
    notifications: {
        email: boolean;
        push: boolean;
        threshold: 'low' | 'medium' | 'high' | 'critical';
    };
    theme: 'light' | 'dark' | 'system';
    language: string;
    timezone: string;
}

export interface ApiError {
    code: string;
    message: string;
    details?: any;
}

export interface WebSocketMessage {
    type: 'event' | 'stats' | 'health' | 'error';
    data: SecurityEvent | ThreatStats | SystemHealth | ApiError;
} 