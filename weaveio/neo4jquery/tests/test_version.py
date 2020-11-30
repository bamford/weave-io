import pytest
import py2neo


def test_update_version_on_first_instance(database: py2neo.Graph, procedure_tag: str):
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        
        WITH specs, child
        // So now there is a multimerged thing
        CALL custom.version{procedure_tag}(specs, child, ['A', 'B'], 'version') YIELD version
        WITH collect(version) as v  // just to finish here
        MATCH (child:MyLabel)
        RETURN child.version
        """
    t = database.run(q).to_table()
    assert len(t) == 1
    assert t[0][0] == 0


def test_dont_update_version_on_duplicate_instance(database: py2neo.Graph, procedure_tag: str):
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        CALL custom.version{procedure_tag}(specs, child, ['A', 'B'], 'version') YIELD version
        
        WITH specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        CALL custom.version{procedure_tag}(specs, child, ['A', 'B'], 'version') YIELD version

        WITH collect(version) as v  // just to finish here
        MATCH (child:MyLabel)
        RETURN child.version
        """
    t = database.run(q).to_table()
    assert len(t) == 1
    assert t[0][0] == 0


def test_dont_update_version_on_diffenent_unversioned_rels(database: py2neo.Graph, procedure_tag: str):
    """Version A parents but not B parents."""
    q = f"""
        UNWIND range(0, 100) as i
        MERGE (a1: A {{id: i}})
        WITH collect(a1) as input_anodes

        MERGE (b1: B {{id: 1}})

        with input_anodes, [b1] as input_bnodes, ['green'] as input_bnames
        CALL apoc.lock.nodes(input_anodes)
        CALL apoc.lock.nodes(input_bnodes)

        UNWIND RANGE(0, SIZE(input_anodes) - 1) AS ai
        WITH [input_anodes[ai], 'arel', {{order: ai}}] AS arow, input_bnodes, input_bnames
        WITH collect(arow) as anodes, input_bnodes, input_bnames

        UNWIND RANGE(0, SIZE(input_bnodes) - 1) AS bi
        WITH [input_bnodes[bi], 'brel', {{order: bi, name: input_bnames[bi]}}] AS brow, anodes
        WITH collect(brow) as bnodes, anodes
        WITH *, bnodes+anodes as specification

        WITH specification as specs
        CALL custom.multimerge{procedure_tag}(specs, ['MyLabel'], {{}}, {{}}, {{}}) YIELD child
        CALL custom.version{procedure_tag}(specs, child, ['A'], 'version') YIELD version

        RETURN child.version
        """
    t = database.run(q).to_table()
    assert len(t) == 1
    assert t[0][0] == 0