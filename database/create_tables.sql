CREATE TABLE usuarios (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    senha_hash TEXT NOT NULL,
    ativo BOOLEAN DEFAULT TRUE,
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE fazendas (
    id SERIAL PRIMARY KEY,              
    codigo TEXT UNIQUE NOT NULL,      
    nome TEXT NOT NULL,
    cidade TEXT,
    estado TEXT,
    area_total_ha NUMERIC
);

CREATE TABLE talhoes (
    id SERIAL PRIMARY KEY,
    codigo TEXT UNIQUE NOT NULL,        -- ex: TAL-001
    fazenda_id INTEGER REFERENCES fazendas(id) ON DELETE CASCADE,
    nome TEXT NOT NULL,
    area_ha NUMERIC,
    cor TEXT
);