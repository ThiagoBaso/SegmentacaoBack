INSERT INTO fazendas (
    codigo, nome, cidade, estado, area_total_ha
) VALUES (
    'FAZ-001',
    'Fazenda Boa Vista',
    'Tupã',
    'SP',
    320.5
);

INSERT INTO talhoes (codigo, fazenda_id, nome, area_ha, cor) VALUES
('TAL-001', (SELECT id FROM fazendas WHERE codigo = 'FAZ-001'), 'Talhão A', 85.2, '#00FFAA'),
('TAL-002', (SELECT id FROM fazendas WHERE codigo = 'FAZ-001'), 'Talhão B', 112.0, '#00AAFF');