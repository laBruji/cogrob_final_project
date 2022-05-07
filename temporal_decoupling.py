from temporal_network import TemporalNetwork, SimpleTemporalConstraint, SimpleContingentTemporalConstraint

ALLOWED_ERROR = 0.0001

class TemporalDecoupling:

    def __init__(self, agent2constraints=None, raw_data=None, objective_value=None, relaxed=False):
        self.init(agent2constraints, raw_data, objective_value, relaxed)

    def init(self, agent2constraints=None, raw_data=None, objective_value=None, relaxed=False):
        # Agent to constraints maps each agent
        # to the set of decoupling constraints
        if agent2constraints is None:
            agent2constraints = {}
        if raw_data is None:
            raw_data = {}
        self.agent2constraints = agent2constraints
        # Raw data is a dictionary with
        # {'u': {(vi, vj): real},
        #  'z': {(vi, vj, vk, vl): boolean},
        #  'c': {(vk, vj, vi): boolean},
        #  'b': {(vi, vj): boolean}
        # }
        self.raw_data = raw_data
        self.objective_value = objective_value

        # Denote if the temporal decoupling is relaxed and only contains necessary constraints for decoupling the external constraints
        self.relaxed = relaxed

    def __repr__(self):
        return "<TemporalDecoupling: {} agents>".format(len(self.agent2constraints))

    def __str__(self):
        return "<TemporalDecoupling: {} agents>".format(len(self.agent2constraints))

    @classmethod
    def from_raw_data(cls, raw_data, ext_reqs, ext_conts, event2agent, objective_value=None):
        proof_reqs, proof_conts = obtain_proof(raw_data, ext_reqs, ext_conts)

        u = raw_data['u']
        z = raw_data['z']
        c = raw_data['c']
        b = raw_data['b']

        contingent_map = {}
        for (vk, vj, vi) in c:
            if c[(vk, vj, vi)] == 1:
                contingent_map[(vk, vj)] = True

        added_map = {}
        agent2constraints = {}

        def add_to_decoupling(vs, ve):
            # If events belong to local agent
            if event2agent(vs) == event2agent(ve) or event2agent(vs) is None or event2agent(ve) is None:
                # If (vs, ve) has not been added
                if vs != ve and (vs, ve) not in added_map:
                    if (vs, ve) in contingent_map:
                        constraint = SimpleContingentTemporalConstraint(vs, ve, -u[(ve, vs)], u[(vs, ve)], 'decoupling:cont({},{})'.format(vs, ve))
                        added_map[(vs, ve)] = True
                        added_map[(ve, vs)] = True
                    elif (ve, vs) in contingent_map:
                        constraint = SimpleContingentTemporalConstraint(ve, vs, -u[(vs, ve)], u[(ve, vs)], 'decoupling:cont({},{})'.format(ve, vs))
                        added_map[(vs, ve)] = True
                        added_map[(ve, vs)] = True
                    else:
                        constraint = SimpleTemporalConstraint(vs, ve, None, u[(vs, ve)], 'decoupling:req({},{})'.format(vs, ve))
                        added_map[(vs, ve)] = True

                    # Add constraint to agent
                    if event2agent(vs) is not None:
                        agent_name = event2agent(vs)
                    else:
                        agent_name = event2agent(ve)
                    assert(agent_name is not None)
                    if agent_name not in agent2constraints:
                        agent2constraints[agent_name] = []
                    agent2constraints[agent_name].append(constraint)


        # Go through all paths to create decoupling constraints
        for (vi, vj) in proof_reqs:
            justification = proof_reqs[(vi, vj)]
            for pair in justification:
                vs, ve = pair[1], pair[2]
                add_to_decoupling(vs, ve)


        for (vi, vj) in proof_conts:
            justification = proof_conts[(vi, vj)]
            for (i, j) in [(vi, vj), (vj, vi)]:
                jus = justification[(i, j)]
                for pair in jus:
                    vs, ve = pair[1], pair[2]
                    add_to_decoupling(vs, ve)
            if len(justification) > 2:
                for (i, j) in justification:
                    if not (i == vi and j == vj) and not (j == vi and i == vj):
                        jus = justification[(i, j)]
                        for pair in jus:
                            vs, ve = pair[1], pair[2]
                            add_to_decoupling(vs, ve)

        return cls(agent2constraints, raw_data, objective_value=objective_value, relaxed=True)

    def pprint(self):
        output_string = "<TemporalDecoupling: {} agents>\n".format(len(self.agent2constraints))
        for agent in self.agent2constraints:
            constraints = self.agent2constraints[agent]
            output_string += "+ Agent {}:\n".format(agent)
            for c in constraints:
                output_string += "  {}\n".format(c)
        output_string += "<TemporalDecoupling/>"
        return output_string

    def add_agent_to_decoupling(self, agent2constraints):
        """
        agent2constraints is type Dictionary
        """
        self.agent2constraints.update(agent2constraints)

    def agent_to_decoupling(self, agent):
        if agent in self.agent2constraints:
            return self.agent2constraints[agent]
        else:
            return []

    def add_raw_data(self, raw_data):
        self.raw_data.update(raw_data)

    def get_raw_data(self):
        return self.raw_data

    def to_json(self, encoder):
        '''
        + 'type': 'temporalDecoupling'
        + 'agent2constraints': {<agent>: constraints}
        + 'objectiveValue': objective_value
        + 'rawData': raw_data
        '''
        return {'type': 'TemporalDecoupling',
                'agent2constraints': encoder.encode(self.agent2constraints),
                'objectiveValue': self.objective_value,
                'rawData': encoder.encode(self.raw_data),
                'relaxed': self.relaxed}

    @classmethod
    def from_json(cls, data, decoder):
        obj = cls()
        decoder.set_id_object(data['$ID'], obj)

        objective_value = data['objectiveValue']
        agent2constraints = data.get('agent2constraints')
        agent2constraints = decoder.decode(agent2constraints)
        raw_data = data.get('rawData')
        raw_data = decoder.decode(raw_data)
        relaxed = data.get('relaxed')
        obj.init(agent2constraints, raw_data, objective_value, relaxed)
        return obj

    def get_relaxed_decoupling(self, ext_reqs, ext_conts):
        """Returns the relaxed decoupling that does not have over-restricting constraints."""
        proof_reqs, proof_conts = obtain_proof(self.raw_data, ext_reqs, ext_conts)

        #  print("# Original decoupling:")
        #  print(self.agent2constraints)

        u = self.raw_data['u']
        z = self.raw_data['z']
        c = self.raw_data['c']
        b = self.raw_data['b']

        useful_pairs = []

        # Pretty print the proof
        for (vi, vj) in proof_reqs:
            justification = proof_reqs[(vi, vj)]
            for pair in justification:
                useful_pairs.append((pair[1], pair[2]))

        for (vi, vj) in proof_conts:
            justification = proof_conts[(vi, vj)]
            for (i, j) in [(vi, vj), (vj, vi)]:
                jus = justification[(i, j)]
                for pair in jus:
                    useful_pairs.append((pair[1], pair[2]))
            if len(justification) > 2:
                for (i, j) in justification:
                    if not (i == vi and j == vj) and not (j == vi and i == vj):
                        jus = justification[(i, j)]
                        for pair in jus:
                            useful_pairs.append((pair[1], pair[2]))

        relaxed_decoupling = TemporalDecoupling(raw_data=self.raw_data.copy(), objective_value=self.objective_value, relaxed=True)

        #  print("Useful Pairs:")
        #  print(useful_pairs)

        agent2constraints = {}
        for agent, constraints in self.agent2constraints.items():
            updated_constraints = []
            for tc in constraints:
                if isinstance(tc, SimpleTemporalConstraint):
                    s = tc.s
                    e = tc.e
                    lb = None
                    ub = None
                    if (s, e) in useful_pairs:
                        ub = tc.ub
                    if (e, s) in useful_pairs:
                        lb = tc.lb
                    if lb is not None or ub is not None:
                        tc_copy = SimpleTemporalConstraint(s, e, lb=lb, ub=ub, name=tc.name)
                        updated_constraints.append(tc_copy)
                elif isinstance(tc, SimpleContingentTemporalConstraint):
                    s = tc.s
                    e = tc.e
                    lb = None
                    ub = None
                    if (s, e) in useful_pairs or (e, s) in useful_pairs:
                        ub = tc.ub
                        lb = tc.lb
                    if lb is not None or ub is not None:
                        tc_copy = SimpleContingentTemporalConstraint(s, e, lb=lb, ub=ub, name=tc.name)
                        updated_constraints.append(tc_copy)
                else:
                    raise ValueError
            agent2constraints[agent] = updated_constraints
        relaxed_decoupling.agent2constraints = agent2constraints
        #  print("# Updated decoupling:")
        #  print(agent2constraints)

        return relaxed_decoupling

    def pprint_proof(self, ext_reqs, ext_conts, warn=False):
        """
        Calls obtain_proof and returns an output string
        """
        proof_reqs, proof_conts = obtain_proof(self.raw_data, ext_reqs, ext_conts, warn=warn)

        u = self.raw_data['u']
        z = self.raw_data['z']
        c = self.raw_data['c']
        b = self.raw_data['b']

        # Pretty print the proof
        output_string = "# Proof for external requirement links:\n"
        for (vi, vj) in proof_reqs:
            justification = proof_reqs[(vi, vj)]
            output_string += "u({},{})({})".format(vi, vj, u[(vi, vj)])
            output_string += " >="
            for pair in justification:
                output_string += " + {}({})".format(pair[0], pair[3])
            output_string += "\n"

        output_string += "\n# Proof for external contingent links:\n"
        for (vi, vj) in proof_conts:
            output_string += "+ ({},{})\n".format(vi, vj)
            justification = proof_conts[(vi, vj)]
            if justification:
                for (i, j) in [(vi, vj), (vj, vi)]:
                    jus = justification[(i, j)]
                    output_string += "u({},{})({})".format(i, j, u[(i, j)])
                    output_string += " <="
                    for pair in jus:
                        output_string += " + {}({})".format(pair[0], pair[3])
                    output_string += "\n"
                if len(justification) > 2:
                    for (i, j) in justification:
                        if not (i == vi and j == vj) and not (j == vi and i == vj):
                            jus = justification[(i, j)]
                            output_string += "Support: u({},{})({})".format(i, j, u[(i, j)])
                            output_string += " >="
                            for pair in jus:
                                output_string += " + {}({})".format(pair[0], pair[3])
                            output_string += "\n"

        return output_string

def obtain_proof(raw_data, ext_reqs, ext_conts, warn=False):
    """Obtain proof of the external constraints decoupled."""

    u = raw_data['u']
    z = raw_data['z']
    c = raw_data['c']
    b = raw_data['b']

    def track_bij_justification(vi, vj):
        # uij >= uik + ukl + ulj
        justification = []
        (bi, bj) = (vi, vj)
        while (bi, bj) in b and b[(bi, bj)] == 1:
            justified = False
            for (zi, zj, zk, zl) in z:
                if zi == bi and zj == bj and z[(zi, zj, zk, zl)] == 1:
                    justification.append(('u({},{})'.format(zi, zk), zi, zk, u[(zi, zk)]))
                    justification.append(('u({},{})'.format(zk, zl), zk, zl, u[(zk, zl)]))
                    justified = True
                    (bi, bj) = (zl, zj)
            assert(justified)
        justification.append(('u({},{})'.format(bi, bj), bi, bj, u[(bi, bj)]))
        return justification

    def track_ckj_justification(vi, vj):
        # ukj >= uki + uij
        # That is, uij <= ukj - uki
        # lkj <= lki + lij
        # That is, ujk >= uik + uji
        # That is, uji <= ujk - uik
        justification = {}
        for (ck, cj, ci) in c:
            if ci == vi and cj == vj and c[(ck, cj, ci)] == 1:
                justification[(vi, vj)] = [('u({},{})'.format(ck, cj), ck, cj, u[(ck, cj)]),
                                            ('-u({},{})'.format(ck, ci), ck, ci, u[(ck, ci)])]
                justification[(vj, vi)] = [('u({},{})'.format(cj, ck), cj, ck, u[(cj, ck)]),
                                            ('-u({},{})'.format(ci, ck), ci, ck, u[(ci, ck)])]
                if (ck, ci) in b and b[(ck, ci)] == 1:
                    # print("int: track_bij_justification({},{})".format(ck, ci))
                    justification[(ck, ci)] = track_bij_justification(ck, ci)
                    # print(justification[(ck, ci)])
                if (ci, ck) in b and b[(ci, ck)] == 1:
                    # print("int: track_bij_justification({},{})".format(ci, ck))
                    justification[(ci, ck)] = track_bij_justification(ci, ck)
                    # print(justification[(ci, ck)])
                return justification
        # No justification is found, possible if using dis search decoupling
        if warn:
            print("WARN: cont link ({}, {}) does not have justification".format(vi, vj))
            return justification
        else:
            print("ERROR: cont link ({}, {}) does not have justification".format(vi, vj))
            raise Exception

    proof_reqs = {}
    # Obtain justification for external requirements
    for link in ext_reqs:
        vi = link.s
        vj = link.e
        if link.ub is not None:
            # Assert that uij needs justification.
            # If not, (vi, vj) may also be a part of a contingent link.
            assert(b[(vi, vj)] == 1)
            # (vi, vj) needs justification
            # print("track_bij_justification({},{})".format(vi, vj))
            proof_reqs[(vi, vj)] = track_bij_justification(vi, vj)
            uij_sum = 0
            for uij in proof_reqs[(vi, vj)]:
                uij_sum += uij[3]
            #  print("({},{}):".format(vi, vj))
            #  print(proof_reqs)
            #  print(u[(vi, vj)])
            #  print(uij_sum)
            assert(u[(vi, vj)] >= uij_sum - ALLOWED_ERROR)
            # print(proof_reqs[(vi, vj)])
        if link.lb is not None:
            assert(b[(vj, vi)] == 1)
            # (vj, vi) needs justification
            # print("track_bij_justification({},{})".format(vj, vi))
            proof_reqs[(vj, vi)] = track_bij_justification(vj, vi)
            uij_sum = 0
            for uij in proof_reqs[(vj, vi)]:
                uij_sum += uij[3]
            #  print("({},{}):".format(vj, vi))
            #  print(proof_reqs)
            #  print(u[(vj, vi)])
            #  print(uij_sum)
            assert(u[(vj, vi)] >= uij_sum - ALLOWED_ERROR)
            # print(proof_reqs[(vj, vi)])

    proof_conts = {}
    # Obtain justification for external contingents
    for link in ext_conts:
        vi = link.s
        vj = link.e
        # (vi, vj) needs justification
        # print("track_ckj_justification({},{})".format(vi, vj))
        proof_conts[(vi, vj)] = track_ckj_justification(vi, vj)
        # print(proof_conts[(vi, vj)])

    return proof_reqs, proof_conts
