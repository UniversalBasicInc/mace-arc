# Fix the "pairs-shifts" typo in the NNPot module (release-2026 branch) and add a regression test.
#   python apply_pair_shifts_fix.py <gromacs-git dir>
import sys, os
R = sys.argv[1]
def edit(path, pairs):
    p = os.path.join(R, path); s = open(p).read()
    for a, b, n in pairs:
        assert s.count(a) == n, (path, a[:70], s.count(a)); s = s.replace(a, b)
    open(p, "w").write(s); print("edited", path)
NN = "src/gromacs/applied_forces/nnpot/"
edit(NN + "nnpot.cpp", [('modelNeedsInput("pairs-shifts")', 'modelNeedsInput("pair-shifts")', 2)])
edit(NN + "tests/nnpot.cpp", [
('''#include "gromacs/mdtypes/imdpoptionprovider_test_helper.h"
''', '''#include "gromacs/mdrunutility/plainpairlistranges.h"
#include "gromacs/mdtypes/imdpoptionprovider_test_helper.h"
''', 1),
('''    void addMdpOptionNNPotActive()
    {
        mdpValueBuilder_.rootObject().addValue("nnpot-active", std::string("true"));
    }
''', '''    void addMdpOptionNNPotActive()
    {
        mdpValueBuilder_.rootObject().addValue("nnpot-active", std::string("true"));
    }

    //! Request the given model inputs and a pair cutoff through the mdp options
    void addMdpOptionModelInputs(const std::vector<std::string>& inputs, real pairCutoff)
    {
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            mdpValueBuilder_.rootObject().addValue("nnpot-model-input" + std::to_string(i + 1), inputs[i]);
        }
        mdpValueBuilder_.rootObject().addValue("nnpot-pair-cutoff", pairCutoff);
    }

    //! Collect the pair-list range the module requests during simulation setup
    std::vector<real> requestedPairlistRanges()
    {
        MDModulesNotifiers notifiers;
        NNPotModule_->subscribeToSimulationSetupNotifications(&notifiers);
        PlainPairlistRanges ranges;
        notifiers.simulationSetupNotifier_.notify(&ranges);
        return { ranges.ranges().begin(), ranges.ranges().end() };
    }
''', 1),
('''TEST_F(NNPotTest, ForceProviderLackingInputThrows)''',
'''//! A model that takes the pair list requests a pair-list range equal to the cutoff
TEST_F(NNPotTest, AtomPairsInputRequestsPairlistRange)
{
    addMdpOptionNNPotActive();
    addMdpOptionModelInputs({ "atom-positions", "atom-numbers", "atom-pairs" }, 0.5);
    makeNNPotModuleWithSetOptions();

    const std::vector<real> ranges = requestedPairlistRanges();
    ASSERT_EQ(ranges.size(), 1U);
    EXPECT_FLOAT_EQ(ranges[0], 0.5);
}

//! A model that takes only the periodic pair shifts needs the pair list just as much
TEST_F(NNPotTest, PairShiftsInputRequestsPairlistRange)
{
    addMdpOptionNNPotActive();
    addMdpOptionModelInputs({ "atom-positions", "atom-numbers", "pair-shifts" }, 0.5);
    makeNNPotModuleWithSetOptions();

    const std::vector<real> ranges = requestedPairlistRanges();
    ASSERT_EQ(ranges.size(), 1U);
    EXPECT_FLOAT_EQ(ranges[0], 0.5);
}

//! Without pair-list inputs no range is requested
TEST_F(NNPotTest, NoPairlistInputRequestsNoRange)
{
    addMdpOptionNNPotActive();
    addMdpOptionModelInputs({ "atom-positions", "atom-numbers" }, 0.5);
    makeNNPotModuleWithSetOptions();

    EXPECT_TRUE(requestedPairlistRanges().empty());
}

TEST_F(NNPotTest, ForceProviderLackingInputThrows)''', 1)])
# release note: 2026.5 bugs-fixed section
p = os.path.join(R, "docs/release-notes/2026/2026.5.rst"); s = open(p).read()
marker = "Fixes where mdrun could behave incorrectly\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
assert marker in s, "2026.5 mdrun section"
s = s.replace(marker, marker + '''
NNPot models requesting only ``pair-shifts`` did not receive a pair list
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

A neural network potential model whose inputs include ``pair-shifts`` but not
``atom-pairs`` was never subscribed to the pair-list construction, because the
option name was misspelled in the check, so the model was passed an empty
shift list.

:issue:`PLACEHOLDER`
''', 1)
open(p, "w").write(s); print("release note 2026.5")
print("FIX-APPLIED")
