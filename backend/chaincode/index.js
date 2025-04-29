'use strict';

const { Contract } = require('fabric-contract-api');

class NeuraShieldContract extends Contract {
    async initLedger(ctx) {
        console.info('============= START : Initialize Ledger ===========');
        console.info('============= END : Initialize Ledger ===========');
    }

    // Create a new model record
    async createModel(ctx, modelId, metadata) {
        console.info('============= START : Create Model ===========');
        
        const model = {
            id: modelId,
            metadata: JSON.parse(metadata),
            status: 'ACTIVE',
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };

        await ctx.stub.putState(modelId, Buffer.from(JSON.stringify(model)));
        console.info('============= END : Create Model ===========');
        return JSON.stringify(model);
    }

    // Query model by ID
    async queryModel(ctx, modelId) {
        console.info('============= START : Query Model ===========');
        const modelAsBytes = await ctx.stub.getState(modelId);
        if (!modelAsBytes || modelAsBytes.length === 0) {
            throw new Error(`Model ${modelId} does not exist`);
        }
        console.info('============= END : Query Model ===========');
        return modelAsBytes.toString();
    }

    // Update model metadata
    async updateModel(ctx, modelId, newMetadata) {
        console.info('============= START : Update Model ===========');
        
        const modelAsBytes = await ctx.stub.getState(modelId);
        if (!modelAsBytes || modelAsBytes.length === 0) {
            throw new Error(`Model ${modelId} does not exist`);
        }
        
        const model = JSON.parse(modelAsBytes.toString());
        model.metadata = JSON.parse(newMetadata);
        model.updatedAt = new Date().toISOString();
        
        await ctx.stub.putState(modelId, Buffer.from(JSON.stringify(model)));
        console.info('============= END : Update Model ===========');
        return JSON.stringify(model);
    }

    // Record model access
    async recordAccess(ctx, modelId, userId, accessType) {
        console.info('============= START : Record Access ===========');
        
        const accessRecord = {
            modelId,
            userId,
            accessType,
            timestamp: new Date().toISOString()
        };
        
        const accessId = `${modelId}_${userId}_${Date.now()}`;
        await ctx.stub.putState(accessId, Buffer.from(JSON.stringify(accessRecord)));
        
        console.info('============= END : Record Access ===========');
        return JSON.stringify(accessRecord);
    }

    // Query access history for a model
    async queryAccessHistory(ctx, modelId) {
        console.info('============= START : Query Access History ===========');
        
        const iterator = await ctx.stub.getStateByPartialCompositeKey('modelId~timestamp', [modelId]);
        const results = [];
        
        while (true) {
            const res = await iterator.next();
            if (res.value && res.value.value.toString()) {
                let record;
                try {
                    record = JSON.parse(res.value.value.toString('utf8'));
                    results.push(record);
                } catch (err) {
                    console.log(err);
                    record = res.value.value.toString('utf8');
                }
            }
            if (res.done) {
                await iterator.close();
                break;
            }
        }
        
        console.info('============= END : Query Access History ===========');
        return JSON.stringify(results);
    }
}

module.exports = NeuraShieldContract; 