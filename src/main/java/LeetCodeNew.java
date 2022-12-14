import com.sun.org.glassfish.gmbal.Description;

import java.util.*;

//public class leetcodenew {
//}
////
//class solution {
//    public int[] shuffle(int[] nums, int n) {
//        if(n==1){
//            return nums;
//        }
//        list<integer> res=new arraylist<integer>(2*n);
//
//        for (int i = 0; i < n; i ++) {
//            res.add(nums[i]);
//            res.add(nums[i+n]);
//        }
//        for (int i = 0; i < 2*n; i++) {
//
//            nums[i]=res.get(i);
//        }
//      return nums;
//    }
//}


//public class treenode {
//    int val;
//    treenode left;
//    treenode right;
//
//    treenode() {
//    }
//
//    treenode(int val) {
//        this.val = val;
//    }
//
//    treenode(int val, treenode left, treenode right) {
//        this.val = val;
//        this.left = left;
//        this.right = right;
//    }
//}

//class solution {
//    public treenode insertintomaxtree(treenode root, int val) {
//        if (root == null) {return new treenode(val);}
//        if (val > root.val) {
//            treenode newroot = new treenode(val);
//            newroot.left = root;
//
//            return newroot;
//        } else {
//            root.right=insertintomaxtree(root.right,val);
//            return root;
//
//        }
//
//    }
//}

//class solution {
//    public boolean validatestacksequences(int[] pushed, int[] popped) {
//        deque<integer> stack = new arraydeque<integer>();
//        int popindex = 0;
//        for (int i = 0; i < pushed.length; i++) {
//            stack.push(pushed[i]);
//            while (!stack.isempty() && stack.peek() == popped[popindex++]) {
//                stack.pop();
//            }
//
//        }
//        return stack.isempty();
//    }
//}

class nummatrix {

    int[][] presum;

    public nummatrix(int[][] matrix) {
        presum = new int[matrix.length + 1][matrix[0].length + 1];
        presum[1][1] = matrix[0][0];
        for (int m1 = 0; m1 < matrix.length; m1++) {
            int sumj = 0;
            for (int m2 = 0; m2 < matrix[0].length; m2++) {
                sumj += matrix[m1][m2];
                presum[m1 + 1][m2 + 1] += presum[m1 + 1][m2] + sumj;
            }
        }
    }

    public int sumregion(int row1, int col1, int row2, int col2) {
        int sumr = 0;
        sumr = presum[row2 + 1][col2 + 1] - presum[row1][col2 + 1] - presum[row2 + 1][col1] + presum[row1][col1];
        return sumr;
    }
}

//
//class solution {
//    public int[] finalprices(int[] prices) {
//        int[] result = new int[prices.length];
//
//        result[prices.length - 1] = prices[prices.length - 1];
//        for (int i = 0; i < prices.length - 1; i++) {
//            boolean flag = true;
//            for (int j = i + 1; j < prices.length; j++) {
//                if (prices[j] <= prices[i]) {
//                    result[i] = prices[i] - prices[j];
//                    flag = false;
//                    break;
//                }
//            }
//            if (flag && result[i] == 0) {
//                result[i] = prices[i];
//            }
//        }
//        return result;
//    }
//}


//public class solution {
//    public boolean hascycle(listnode head) {
//        listnode slow = head;
//        listnode fast = head;
//        while (fast!= null && fast != null) {
//            fast = fast.next.next;
//            slow = slow.next;
//            if (fast == slow) {
//                return true;
//            }
//        }
//        return false;
//    }
//}


//class solution {
//    public boolean canpartition(int[] nums) {
//        int sum = 0;
//
//        for (int num : nums) {
//            sum += num;
//        }
//        if ((sum & 1) == 0) {
//            return false;
//        }
//        int w = sum >> 1;
//        int[] dp = new int[w + 1];
//        dp[0] = 1;
//        for (int num : nums) {
//            for (int weight = w; weight >= num; weight--) {
//                dp[weight] += dp[weight - num];
//            }
//        }
//        return dp[w] != 0;
//    }
//}
//class solution {
//
//    private int maxl = 0;
//
//    public int longestunivaluepath(treenode root) {
//        /**
//         ?????????????????????124???, ????????????????????????, ???????????????????????????????????????, ??????????????????????????????:
//         1. ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//         2. ??????????????????????????????????????????????????????, ????????????????????????????????????
//         ??????????????????????????????, ???????????????????????????????????????????????????????????????????????????????????????????????????
//         **/
//        if(root == null) return 0;
//        getmaxl(root, root.val);
//        return maxl;
//    }
//
//    private int getmaxl(treenode r, int val) {
//        if(r == null) return 0;
//        int left = getmaxl(r.left, r.val);
//        int right = getmaxl(r.right, r.val);
//        maxl = math.max(maxl, left+right); // ???????????????????????????1??????????????????1
//        if(r.val == val) // ????????????????????????????????????????????????????????????????????????????????????, ????????????0
//            return math.max(left, right) + 1;
//        return 0;
//    }
//}


//class solution {
//    public int[] twosum(int[] nums, int target) {
//        int n = nums.length;
//        if (n < 2) {
//            return new int[2];
//        }
//        int left = 0;
//        int right = n - 1;
//        while (left < right) {
//            if (nums[left] + nums[right] == target) {
//                return new int[]{nums[left], nums[right]};
//            } else if (nums[left] + nums[right] > target) {
//                right--;
//            } else if (nums[left] + nums[right] < target) {
//                left++;
//            }
//        }
//        return new int[2];
//    }
//}

//class solution {
//    public int findlongestchain(int[][] pairs) {
//        arrays.sort(pairs,(a,b)-> a[1]-b[1]);
//        int res = 1,tmp = pairs[0][1];
//        for(int i = 1;i < pairs.length;i++){
//            if(pairs[i][0] > tmp){
//                res++;
//                tmp = pairs[i][1];
//            }
//        }
//        return res;
//    }
//}

//class solution {
//    public listnode reverselist(listnode head) {
//        if (head == null) {
//            return head;
//        }
//        if (head.next == null) {
//            return head;
//        }
//        listnode pre = new listnode(0, head);
//
//        listnode cur = head;
//        while (cur != null) {
//            listnode next = cur.next;
//            cur.next = pre;
//            pre = cur;
//            cur = next;
//        }
//        head.next = null;
//        return pre;
//    }
//}
//
//class solution {
//    public listnode reverselist(listnode head) {
//        if (head == null) {
//            return head;
//        }
//        if (head.next == null) {
//            return head;
//        }
//        listnode res = reverselist(head.next);
//        head.next.next = head;
//        head.next = null;
//        return res;
//    }
//}


//class solution {
//    public int foursumcount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
//        map<integer, integer> map12 = new hashmap<integer, integer>();
//        map<integer, integer> map34 = new hashmap<integer, integer>();
//        arrays.sort(nums1);
//        arrays.sort(nums2);
//        arrays.sort(nums3);
//        arrays.sort(nums4);
//        int count = 0;
//        for (int i = 0; i < nums1.length; i++) {
//            for (int j = 0; j < nums2.length; j++) {
//                map12.put(nums1[i] + nums2[j], map12.getordefault(nums1[i] + nums2[j], 0) + 1);
//            }
//        }
//        for (int i = 0; i < nums3.length; i++) {
//            for (int j = 0; j < nums4.length; j++) {
//                map34.put(nums3[i] + nums4[j], map34.getordefault(nums3[i] + nums4[j], 0) + 1);
//            }
//        }
//        for (int ab : map12.keyset()) {
//            count += map12.get(ab) * map34.getordefault(-ab, 0);
//        }
//        return count;
//    }
//}
//class solution {
//    public list<list<integer>> threesum(int[] nums) {
//        arrays.sort(nums);
//        list<list<integer>> result = new arraylist<list<integer>>;
//        for (int i = 0; i < nums.length - 2; i++) {
//            list<integer> tmp = new arraylist<integer>();
//            int l = i + 1, r = nums.length - 1;
//            while (l < r) {
//                int sum = nums[i] + nums[r] + nums[l];
//                if (sum < 0) {
//                    l++;
//                } else if (sum > 0) {
//                    r--;
//                } else {
//                    tmp.add(nums[i]);
//                    tmp.add(nums[l]);
//                    tmp.add(nums[r]);
//                    if (!result.contains(tmp)) {
//                        result.add(tmp);
//                    }
//                    l++;
//                    r--;
//                }
//
//            }
//        }
//        return result;
//    }
//}
//
//
//class solution {
//    public int numspecial(int[][] mat) {
//        int res = 0;
//        for (int i = 0; i < mat.length; i++) {
//            for (int j = 0; j < mat[i].length; j++) {
//                if (mat[i][j] == 1 && isspecial(mat, i, j)) {
//                    res++;
//                }
//            }
//        }
//        return res;
//    }
//
//    private boolean isspecial(int[][] mat, int row, int col) {
//        int count1 = 0, count2 = 0;
//        for (int i = 0; i < mat.length; i++) {
//            if (mat[i][col] == 1) {
//                count1++;
//            }
//        }
//        for (int j = 0; j < mat[row].length; j++) {
//            if (mat[row][j] == 1) {
//                count2++;
//            }
//        }
//        return count1 == 1 && count2 == 1;
//    }
//}
//
//class solution {
//    map<string, integer> m = new hashmap<>();
//    list<treenode> ans = new arraylist<>();
//
//    public list<treenode> findduplicatesubtrees(treenode root) {
//        dfs(root);
//        return ans;
//    }
//
//    string dfs(treenode root) {
//        if (root == null) {
//            return "#";
//        }
//        string s = string.valueof(root.val) + "," + dfs(root.left) + "," + dfs(root.right);
//        int n = m.getordefault(s, 0);
//        if (n == 1) {
//            ans.add(root);
//        }
//        m.put(s, n + 1);
//        return s;
//    }
//}
//
//class solution {
//    public string reorderspaces(string text) {
//        int countblanks = 0;
//        stringbuilder sb = new stringbuilder(text.length());
//        for (int i = 0; i < text.length(); i++) {
//            if (text.charat(i) == ' ') {
//                countblanks++;
//            }
//        }
//        string[] words = text.trim().split("\\s+");
//        if (words.length == 1) {
//            for (int i = 0; i < countblanks; i++) {
//                words[0] += ' ';
//            }
//            return words[0];
//        }
//        int mean = countblanks / (words.length - 1);
//        for (int i = 0; i < words.length; i++) {
//
//            sb.append(words[i]);
//            if (countblanks > mean) {
//                for (int j = 0; j < mean; j++) {
//                    sb.append(' ');
//                }
//                countblanks -= mean;
//            } else {
//                for (int j = 0; j < countblanks; j++) {
//                    sb.append(' ');
//                }
//                countblanks=0;
//            }
//        }
//        if(countblanks!=0){
//            for (int j = 0; j < countblanks; j++) {
//                sb.append(' ');
//            }
//        }
//        return sb.tostring();
//    }
//}class solution {
//    public int[] constructarray(int n, int k) {
//        int[] res = new int[n];
//        int odd = 1, even = k+1;
//        //?????????[0, k]????????????????????????[1,2,3..]
//        for(int i = 0; i <= k; i+=2){
//            res[i] = odd++;
//        }
//        //?????????[0, k]????????????????????????[k + 1, k, k - 1...]
//        for(int i = 1; i <= k; i+=2){
//            res[i] = even--;
//        }
//        //?????????[k + 1, n - 1]??????????????????
//        for(int i = k+1; i < n; ++i){
//            res[i] = i+1;
//        }
//        return res;
//    }
//}


//class solution {
//
//
//    public int specialarray(int[] nums) {
//        int n = nums.length;
//        if (n == 1) {
//            return nums[0] >= 1 ? 1 : -1;
//        }
//        arrays.sort(nums);
//
//        map<integer, integer> map = new hashmap<>();
//
//        //??????????????????
//        for (int i = 0; i < n; i++) {
//            map.put(nums[i], math.max(map.getordefault(nums[i], n - i), n - i));
//            if (nums[i] == map.get(nums[i])) {
//                return nums[i];
//            }
//        }
//        //??????????????????
//        if (nums[0] >= n) {
//            return n;
//        }
//        for (int i = 1; i <= n; i++) {
//            for (int j = 1; j < n; j++) {
//                if (i <= nums[j] && i > nums[j - 1]) {
//                    if (i == map.get(nums[j])) {
//                        return i;
//                    }
//                }
//            }
//        }
//        return -1;
//    }
//}
//
//
//class Solution {
///*
// * @description:
// * @author: ChenSC
// * @date: 2022/9/13 16:57
// * @param:
// * @param: num
// * @return: int
// **/
//    public int maximumSwap(int num) {
//        String str = Integer.toString(num);
//        int[] nums = new int[str.length()];
//        int[] tmp = new int[str.length()];
//        for (int i = 0; i < nums.length; i++) {
//            nums[i] = str.charAt(i) - '0';
//            tmp[i] = nums[i];
//        }
//        Arrays.sort(tmp);
//        boolean flag = false;
//        for (int i = 0; i < nums.length - 1; i++) {
//            if (nums[i] != tmp[nums.length - 1 - i]) {
//                for (int j = nums.length - 1; j > i; j--) {
//                    if (nums[j] == tmp[nums.length - 1 - i]) {
//                        int t = nums[j];
//                        nums[j] = nums[i];
//                        nums[i] = t;
//                        flag = true;
//                        break;
//                    }
//                }
//            }
//
//            if (flag) {
//                break;
//            }
//        }
//        int res = 0;
//        for (int i = 0; i < nums.length; i++) {
//            res += nums[i] * Math.pow(10, nums.length - i - 1);
//        }
//        return res;
//    }
//}
//
//
//class Solution {
//    public static void main(String[] args) {
//        int[] nums = {9, 7, 8, 7, 7, 8, 4, 4, 6, 8, 8, 7, 6, 8, 8, 9, 2, 6, 0, 0, 1, 10, 8, 6, 3, 3, 5, 1, 10, 9, 0, 7, 10, 0, 10, 4, 1, 10, 6, 9, 3, 6, 0, 0, 2, 7, 0, 6, 7, 2, 9, 7, 7, 3, 0, 1, 6, 1, 10, 3};
//        int[] test = nums.clone();
//        Arrays.sort(nums);
//        System.out.println(nums);
//        System.out.println(test);
//        for (int num : nums) {
//            System.out.print(num + " ");
//        }
//        System.out.println();
//        for (int num : test) {
//            System.out.print(num + " ");
//        }
//        System.out.println();
//
//    }
//
//
//    public double trimMean(int[] arr) {
//        int n = arr.length;
//        Arrays.sort(arr);
//        int exLen = (int) (n * 0.05);
//        double sum = 0;
//        for (int i = exLen; i < n - exLen; i++) {
//            sum += arr[i];
//        }
//        return sum / (n - 2 * exLen);
//    }
//
//
//}
//
//
//class Solution {
//    public int[] missingTwo(int[] nums) {
//        int n = nums.length;
//        int[] ret = new int[]{n + 1, n + 2};
//        for (int i = 0; i < n; i++) {
//            int cur = nums[i];
//            while (cur <= n && cur != nums[cur - 1]) {
//                int next = nums[cur - 1];
//                nums[cur - 1] = cur;
//                cur = next;
//            }
//            if (cur == n + 1) ret[0] = -1;
//            else if (cur == n + 2) ret[1] = -1;
//        }
//        int i = 0;
//        for (int j = 0; j < 2; j++) {
//            if (ret[j] == -1)
//                for (; i < n; i++) {
//                    if (nums[i] != i + 1) {
//                        ret[j] = ++i;
//                        break;
//                    }
//                }
//        }
//        return ret;
//    }
//}
//
//class Solution {
//    public int[] missingTwo(int[] nums) {
//        int xorsum = 0;
//        int n = nums.length + 2;
//        for (int num : nums) {
//            xorsum ^= num;
//        }
//        for (int i = 1; i <= n; i++) {
//            xorsum ^= i;
//        }
//        // ????????????
//        int lsb = (xorsum == Integer.MIN_VALUE ? xorsum : xorsum & (-xorsum));
//        int type1 = 0, type2 = 0;
//        for (int num : nums) {
//            if ((num & lsb) != 0) {
//                type1 ^= num;
//            } else {
//                type2 ^= num;
//            }
//        }
//        for (int i = 1; i <= n; i++) {
//            if ((i & lsb) != 0) {
//                type1 ^= i;
//            } else {
//                type2 ^= i;
//            }
//        }
//        return new int[]{type1, type2};
//    }
//}
//
//class Solution {
//    public int[] missingTwo(int[] nums) {
//        Arrays.sort(nums);
//        List<Integer> total = new ArrayList<Integer>(nums.length + 2);
//        for (int i = 0; i < nums.length + 2; i++) {
//            total.add(i + 1);
//        }
//        for (int num : nums) {
//            total.remove(Integer.valueOf(num));
//        }
//        return new int[]{total.get(0), total.get(1)};
//    }
//
//    public static void main(String[] args) {
//
//
//        Integer a = 1000, b = 1000;
//        System.out.println(a == b);
//
//    }
//
//
//
//
//}
//
//class Solution {
//    public boolean CheckPermutation(String s1, String s2) {
//        if (s1.length() != s2.length()) {
//            return false;
//        }
//        char[] abc1 = s1.toCharArray();
//        char[] abc2 = s2.toCharArray();
//        Arrays.sort(abc1);
//        Arrays.sort(abc2);
//        for (int i = 0; i < abc1.length; i++) {
//            if (abc1[i] != abc2[i]) {
//                return false;
//            }
//        }
//        return true;
//    }
//}
//class Solution {
//    public boolean CheckPermutation(String s1, String s2) {
//        int sum1 = 0;
//        int sum2 = 0;
//        if(s1.length() != s2.length()) return false;
//        else{
//            for(int i=0; i<s1.length(); i++){
//                sum1 += s1.charAt(i);
//                sum2 += s2.charAt(i);
//            }
//        }
//        return (sum1 == sum2);
//    }
//}


  /Definition for singly-linked list.

class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}

class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
//?????????????????????
        List<Integer> tmp = new ArrayList<Integer>();
        for (int i = 0; i < lists.length; i++) {
            ListNode cur = lists[i];
            while (cur != null) {
                tmp.add(cur.val);
                cur = cur.next;
            }
        }
        tmp.sort(Comparator.naturalOrder());
        ListNode dummy = new ListNode(0);
        ListNode p = dummy;
        for (int i = 0; i < tmp.size(); i++) {
            p.next = new ListNode(tmp.get(i));
            p = p.next;

        }
        return dummy.next;
    }

}

class LevelTrverse {
    class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        public TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }

        public TreeNode(int val) {
            this.val = val;
        }
    }

    void help(TreeNode root) {
        if (root == null) {
            return;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int dx = queue.size();
            for (int i = 0; i < dx; i++) {
                TreeNode cur = queue.poll();
                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }

            }
        }

    }
}

class Solution {
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        // ??????????????? amount + 1?????????????????? amount + 1
        Arrays.fill(dp, amount + 1);

        // base case
        dp[0] = 0;
        // ?????? for ??????????????????????????????????????????
        for (int i = 0; i < dp.length; i++) {
            // ?????? for ????????????????????????????????????
            for (int coin : coins) {
                // ????????????????????????
                if (i - coin < 0) {
                    continue;
                }
                dp[i] = Math.min(dp[i], 1 + dp[i - coin]);
            }
        }
        return (dp[amount] == amount + 1) ? -1 : dp[amount];
    }
}

class Solution {
    List<List<Integer>> res = new ArrayList<>();

    public List<List<Integer>> permute(int[] nums) {

        //List<Integer> track = new LinkedList<>();
        LinkedList<Integer> track = new LinkedList<>();
        if (nums.length == 0) {
            return res;
        }
        boolean[] used = new boolean[nums.length];
        backtrack(nums, track, used);
        return res;
    }

    void backtrack(int[] nums, LinkedList<Integer> track, boolean[] used) {
        if (track.size() == nums.length) {
            res.add(new LinkedList<>(track));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            used[i] = true;
            track.add(nums[i]);
            backtrack(nums, track, used);

            //track.remove(track.size() - 1);
            track.removeLast();
            used[i] = false;
        }
    }
}


class Solution {
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 0; i < dp.length; i++) {
            for (int coin : coins) {
                if (i - coin < 0) {
                    continue;
                }
                dp[i] = Math.min(dp[i], 1 + dp[i - coin]);

            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }

}


class Solution {

    List<List<String>> res = new ArrayList<List<String>>();

    public List<List<String>> solveNQueens(int n) {
        char[][] board = new char[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] = '.';
            }
        }
        boolean[][] isValid = new boolean[n][n];
        backtrack(board, 0);
        return res;
    }

    void backtrack(char[][] board, int row) {
        int n = board.length;
        List<String> track = new ArrayList<>();
        if (row == board.length) {
            for (int i = 0; i < n; i++) {
                StringBuilder sb = new StringBuilder(n);
                for (int j = 0; j < n; j++) {
                    sb.append(board[i][j]);
                }
                track.add(sb.toString());
            }
            res.add(track);
            return;
        }

        for (int col = 0; col < n; col++) {
            if (isValid(board, row, col)) {
                board[row][col] = 'Q';
                backtrack(board, row + 1);
                board[row][col] = '.';
            }
        }
    }

    boolean isValid(char[][] board, int row, int col) {
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') {
                return false;
            }
        }
        int n = board.length;
        for (int i = row - 1, j = col + 1;
             i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        // ??????????????????????????????????????????
        for (int i = row - 1, j = col - 1;
             i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }
}


class Solution {
    public int getKthMagicNumber(int k) {
        int[] res = new int[k];
        res[0] = 1;
        int p3 = 0, p5 = 0, p7 = 0;
        for (int i = 1; i < k; i++) {
            int tmp = Math.min(Math.min(res[p3] * 3, res[p5] * 5), res[p7] * 7);
            if (tmp % 3 == 0) {
                p3++;
            }
            if (tmp % 5 == 0) {
                p5++;
            }
            if (tmp % 7 == 0) {
                p7++;
            }
            res[i] = tmp;
        }

        return res[k - 1];
    }
}

/*
 * @description: ???????????????????????????
 * @author: ChenSC
 * @date: 2022/9/28 14:32
 * @param:
 * @param: null
 * @return: null
 **/
class Solution {
    List<List<Integer>> res = new ArrayList<List<Integer>>();
    List<Integer> track = new LinkedList<>();

    public List<List<Integer>> permuteUnique(int[] nums) {
        if (nums.length == 0) {
            return res;
        }
        boolean[] used = new boolean[nums.length];
        backtrack(nums, used);
        return res;
    }

    void backtrack(int[] nums, boolean[] used) {
        if (track.size() == nums.length) {
            res.add(new LinkedList<>(track));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (!used[i]) {
                track.add(nums[i]);
                used[i] = true;
                backtrack(nums, used);
                used[i] = false;
                track.remove(track.size() - 1);
            }
        }
    }
}

/*
 * @description:  ?????????II ????????????????????????
 * @author: ChenSC
 * @date: 2022/9/28 14:49
 * @param:
 * @param: null
 * @return: null
 **/
class Solution {
    List<List<Integer>> res = new ArrayList<List<Integer>>();
    List<Integer> track = new LinkedList<>();

    public List<List<Integer>> permuteUnique(int[] nums) {
        if (nums.length == 0) {
            return res;
        }
        Arrays.sort(nums);
        boolean[] used = new boolean[nums.length];
        backtrack(nums, used);
        return res;
    }

    void backtrack(int[] nums, boolean[] used) {
        if (track.size() == nums.length) {
            res.add(new LinkedList<>(track));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }
            track.add(nums[i]);
            used[i] = true;
            backtrack(nums, used);
            used[i] = false;
            track.remove(track.size() - 1);
        }
    }
}


/*
 * @description:  ?????????????????????????????????
 * @author: ChenSC
 * @date: 2022/9/28 14:33
 * @param:
 * @param: null
 * @return: null
 **/
class Solution {
    List<List<Integer>> res = new ArrayList<List<Integer>>();
    List<Integer> track = new LinkedList<>();

    public List<List<Integer>> permuteUnique(int[] nums) {
        if (nums.length == 0) {
            return res;
        }
        backtrack(nums, 0);
        return res;
    }

    void backtrack(int[] nums, int indx) {
        res.add(new LinkedList<>(track));

        for (int i = indx; i < nums.length; i++) {
            track.add(nums[i]);
            backtrack(nums, i + 1);
            track.remove(track.size() - 1);
        }
    }
}


/*
 * @description:  ???????????? ??????????????????
 * @author: ChenSC
 * @date: 2022/9/28 15:12
 * @param:
 * @param: null
 * @return: null
 **/
class Solution {
    List<List<Integer>> res = new ArrayList<List<Integer>>();
    List<Integer> track = new LinkedList<>();

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        if (nums.length == 0) {
            return res;
        }
        Arrays.sort(nums);//?????????????????????????????????
        backtrack(nums, 0);
        return res;
    }

    void backtrack(int[] nums, int indx) {
        res.add(new LinkedList<>(track));

        for (int i = indx; i < nums.length; i++) {

            if (i > indx && nums[i] == nums[i - 1]) {
                continue;
            }
            track.add(nums[i]);
            backtrack(nums, i + 1);
            track.remove(track.size() - 1);
        }
    }
}


/*
 * @description:  ????????????target?????????
 * @author: ChenSC
 * @date: 2022/9/28 15:31
 * @param:
 * @param: null
 * @return: null
 **/
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    // ?????????????????????
    LinkedList<Integer> track = new LinkedList<>();
    // ?????? track ??????????????????
    int trackSum = 0;

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        if (candidates.length == 0) {
            return res;
        }
        // ??????????????????????????????????????????
        Arrays.sort(candidates);
        backtrack(candidates, 0, target);
        return res;
    }

    // ?????????????????????
    void backtrack(int[] nums, int start, int target) {
        // base case????????????????????????????????????????????????
        if (trackSum == target) {
            res.add(new LinkedList<>(track));
            return;
        }
        // base case?????????????????????????????????
        if (trackSum > target) {
            return;
        }

        // ????????????????????????
        for (int i = start; i < nums.length; i++) {
            // ??????????????????????????????????????????????????????
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            // ?????????
            track.add(nums[i]);
            trackSum += nums[i];
            // ??????????????????????????????
            backtrack(nums, i + 1, target);
            // ????????????
            track.removeLast();
            trackSum -= nums[i];
        }
    }
}

/*
 * @description:????????????target?????????
 * @author: ChenSC
 * @date: 2022/9/28 16:08
 * @param:
 * @param: null
 * @return: null
 **/
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    List<Integer> track = new LinkedList<>();
    int trackSum;

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        if (candidates.length == 0) {
            return res;
        }
        Arrays.sort(candidates);
        backtrack(candidates, 0, target);
        return res;
    }

    void backtrack(int[] candidates, int indx, int target) {
        if (trackSum == target) {
            res.add(new LinkedList<>(track));
            return;
        }
        for (int i = indx; i < candidates.length; i++) {
            if (candidates[i] <= target) {
                if (i > indx && candidates[i] == candidates[i - 1]) {
                    continue;
                }
            }

            if (trackSum + candidates[i] <= target) {//???????????????
                track.add(candidates[i]);
                trackSum += candidates[i];
                backtrack(candidates, i + 1, target);
                track.remove(track.size() - 1);
                trackSum -= candidates[i];
            } else {
                break;
            }
        }
    }
}

/*
 * @description:  ????????????????????????????????????
 * @author: ChenSC
 * @date: 2022/9/28 16:20
 * @param:
 * @param: null
 * @return: null
 **/
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> track = new LinkedList<>();
    int trackSum;

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates.length == 0) {
            return res;
        }

        Arrays.sort(candidates);
        backtrack(candidates, 0, target);
        return res;
    }

    void backtrack(int[] candidates, int indx, int target) {
        if (trackSum == target) {
            res.add((new ArrayList<>(track)));
            return;
        }
        if (trackSum > target) {
            return;
        }
        for (int i = indx; i < candidates.length; i++) {
            track.add(candidates[i]);
            trackSum += candidates[i];
            backtrack(candidates, i, target);
            track.remove(track.size() - 1);
            trackSum -= candidates[i];

        }
    }
}
//class Solution {
//    public List<List<Integer>> combinationSum(int[] candidates, int target) {
//        int[] current = new int[1000];
//        List<List<Integer>> result = new ArrayList<>();
//        Arrays.sort(candidates);
//        dfs(result, candidates, target, current, 0, candidates.length - 1);
//        return result;
//    }
//
//    private void dfs(List<List<Integer>> result, int[] candidates, int target, int[] current, int depth, int next) {
//        if (target == 0) {
//            List<Integer> temp = new ArrayList<>();
//            for (int i = 0;i < depth;i++) {
//                temp.add(current[i]);
//            }
//            result.add(temp);
//            return;
//        }
//        for (int i = next;i >= 0;i--) {
//            if (target >= candidates[i]) {
//                current[depth] = candidates[i];
//                dfs(result, candidates, target - candidates[i], current, depth + 1, i);
//            }
//        }
//    }
//
//    public static void main(String[] args) {
//        int[] num = {2,3,6,7};
//        Solution solution = new Solution();
//        List<List<Integer>> result = solution.combinationSum(num, 7);
//        for (List<Integer> list : result) {
//            for (Integer ele : list) {
//                System.out.print(ele + " ");
//            }
//            System.out.println("");
//        }
//    }
//}

/*
 * @description:  ????????????????????? ??????
 * @author: ChenSC
 * @date: 2022/9/28 16:22
 * @param:
 * @param: null
 * @return: null
 **/

class Solution {
    List<List<Integer>> res = new LinkedList<>();
    LinkedList<Integer> track = new LinkedList<>();

    public List<List<Integer>> permuteRepeat(int[] nums) {
        backtrack(nums);
        return res;
    }

    // ????????????????????????
    void backtrack(int[] nums) {
        // base case?????????????????????
        if (track.size() == nums.length) {
            // ???????????????????????????
            res.add(new LinkedList(track));
            return;
        }

        // ????????????????????????
        for (int i = 0; i < nums.length; i++) {
            // ?????????
            track.add(nums[i]);
            // ????????????????????????
            backtrack(nums);
            // ????????????
            track.removeLast();
        }
    }
}


/**
 * @description: BFS ????????????
 * @param:
 * @return:
 * @author: ChenSC
 * @time: 2022/9/28 16:51
 */
class Solution {

    public int openLock(String[] deadends, String target) {
        int count = 0;
        Queue<String> queue = new LinkedList<String>();

        Set<String> set = new HashSet<String>();
        Set<String> visited = new HashSet<>();
        for (String str : deadends) {
            set.add(str);
        }
        queue.offer("0000");
        visited.add("0000");
        while (!(queue.isEmpty())) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                String cur = queue.poll();
                if (cur.equals(target)) {
                    return count;
                }
                if (set.contains(cur)) {
                    continue;
                }
                for (int j = 0; j < 4; j++) {
                    String up = plusOne(cur, j);
                    String down = minusOne(cur, j);
                    if (!visited.contains(up)) {
                        queue.offer(up);
                        visited.add(up);
                    }
                    if (!visited.contains(down)) {
                        queue.offer(down);
                        visited.add(down);
                    }
                }

            }
            count++;
        }
        return -1;

    }

    int openLock1(String[] deadends, String target) {
        Set<String> deads = new HashSet<>();
        for (String s : deadends) deads.add(s);
        // ????????????????????????????????????????????????????????????
        Set<String> q1 = new HashSet<>();
        Set<String> q2 = new HashSet<>();
        Set<String> visited = new HashSet<>();

        int step = 0;
        q1.add("0000");
        q2.add(target);

        while (!q1.isEmpty() && !q2.isEmpty()) {
            // ??????????????????????????????????????????????????? temp ??????????????????
            Set<String> temp = new HashSet<>();

            /* ??? q1 ????????????????????????????????? */
            for (String cur : q1) {
                /* ???????????????????????? */
                if (deads.contains(cur))
                    continue;
                if (q2.contains(cur))
                    return step;

                visited.add(cur);

                /* ??????????????????????????????????????????????????? */
                for (int j = 0; j < 4; j++) {
                    String up = plusOne(cur, j);
                    if (!visited.contains(up))
                        temp.add(up);
                    String down = minusOne(cur, j);
                    if (!visited.contains(down))
                        temp.add(down);
                }
            }
            /* ????????????????????? */
            step++;
            // temp ????????? q1
            // ???????????? q1 q2???????????? while ???????????? q2
            q1 = q2;
            q2 = temp;
        }
        return -1;
    }

    String plusOne(String str, int i) {
        char[] chars = str.toCharArray();
        if (chars[i] == '9') {
            chars[i] = '0';
        } else {
            chars[i]++;
        }
        return new String(chars);
    }

    String minusOne(String str, int i) {
        char[] chars = str.toCharArray();
        if (chars[i] == '0') {
            chars[i] = '9';
        } else {
            chars[i]--;
        }
        return new String(chars);
    }

}


/*
 * @description:
 * @author: ChenSC
 * @date: 2022/9/28 20:51
 * @param:
 * @param: null
 * @return: null
 **/
class Solution {
    public String minWindow(String s, String t) {
        Map<Character, Integer> window = new HashMap<>();
        Map<Character, Integer> need = new HashMap<>();
        char[] cht = t.toCharArray();
        for (char c : cht) {
            need.put(c, need.getOrDefault(c, 0) + 1);
        }
        int valid = 0;
        int left = 0;
        int right = 0;
        int start = 0;
        int len = Integer.MAX_VALUE;
        while (right < s.length()) {
            char c = s.charAt(right);
            right++;
            if (need.get(c) != null) {
                window.put(c, window.getOrDefault(c, 0) + 1);
                if (window.get(c).equals(need.get(c))) {
                    valid++;
                }
            }
            while (valid == need.size()) {
                if (right - left < len) {
                    start = left;
                    len = right - left;
                }
                char d = s.charAt(left);
                left++;
                if (need.get(d) != null) {
                    if (window.get(d).equals(need.get(d))) {
                        valid--;
                    }
                    window.put(d, window.getOrDefault(d, 0) - 1);
                }
            }
        }
        return len == Integer.MAX_VALUE ? "" : s.substring(start, start + len);
    }
}

class Solution {
    public boolean checkInclusion(String s1, String s2) {
        Map<Character, Integer> window = new HashMap<Character, Integer>();
        Map<Character, Integer> need = new HashMap<Character, Integer>();
        for (char c : s1.toCharArray()) {
            need.put(c, need.getOrDefault(c, 0) + 1);
        }
        int left = 0;
        int right = 0;
        int valid = 0;
        while (right < s2.length()) {
            char c = s2.charAt(right);
            right++;
            if (need.get(c) != null) {
                window.put(c, window.getOrDefault(c, 0) + 1);
                if (window.get(c).equals(need.get(c))) {
                    valid++;
                }
            }
            while (right - left >= s1.length()) {
                if (valid == need.size() && right - left == s2.length()) {
                    return true;
                }
                char d = s2.charAt(left);
                left++;
                if (need.get(d) != null) {
                    if (window.get(d).equals(need.get(d))) {
                        valid--;
                    }
                    window.put(d, window.getOrDefault(d, 0) - 1);
                }
            }
        }
        return false;
    }
}

class Solution {
    public boolean checkInclusion(String s1, String s2) {
        // ????????????,???????????????????????????????????????????????????s1????????????????????????????????????????????????????????????s1??????????????????????????????
        HashMap<Character, Integer> need = new HashMap<>();
        HashMap<Character, Integer> window = new HashMap<>();
        for (int i = 0; i < s1.length(); i++) {
            char ch = s1.charAt(i);
            need.put(ch, need.getOrDefault(ch, 0) + 1);
        }
        int left = 0, right = 0; // ?????????????????????
        int valid = 0;  // ???????????????s1?????????????????????
        while (right < s2.length()) {
            char c = s2.charAt(right);
            ++right; // ????????????
            if (need.containsKey(c)) {
                window.put(c, window.getOrDefault(c, 0) + 1);
                if (window.get(c).equals(need.get(c))) { // ??????c??????s1?????????
                    valid++;
                }
            }
            // ????????????????????????[left, right)??????????????????????????????????????????????????????????????????s1.length()
            if (right - left == s1.length()) {  // ???????????????????????????????????????
                if (valid == need.size()) { // ??????????????????s1???????????????
                    return true;
                }
                // ????????????
                char d = s2.charAt(left);
                ++left;
                if (need.containsKey(d)) {
                    if (window.get(d).equals(need.get(d))) { // ?????????d?????????-1?????????????????????????????????-1
                        valid--;
                    }
                    window.put(d, window.get(d) - 1); // ??????d?????????????????????-1
                }
            }
        }
        return false;
    }
}

class Solution {
    public boolean checkInclusion(String s1, String s2) {
        // ?????????
        int[] word = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            char c = s1.charAt(i);
            word[c - 'a']++;
        }

        // ????????????
        for (int i = 0, j = 0; i < s2.length(); i++) {
            // ??????
            word[s2.charAt(i) - 'a']--;

            // ??????
            while (word[s2.charAt(i) - 'a'] < 0) {
                word[s2.charAt(j) - 'a']++;
                j++;
            }

            // ??????
            if (i - j + 1 == s1.length()) {
                return true;
            }
        }

        return false;
    }
}

class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        Map<Character, Integer> window = new HashMap<>();
        Map<Character, Integer> need = new HashMap<>();
        List<Integer> res = new ArrayList<>();
        for (char c : p.toCharArray()) {

            need.put(c, need.getOrDefault(c, 0) + 1);

        }
        int left = 0, right = 0;
        int valid = 0;
        while (right < s.length()) {
            char c = s.charAt(right);
            right++;
            if (need.containsKey(c)) {
                window.put(c, window.getOrDefault(c, 0) + 1);
                if (window.get(c).equals(need.get(c))) {
                    valid++;
                }
            }
            if (right - left == p.length()) {
                if (valid == need.size()) {
                    res.add(left);
                }
                char d = s.charAt(left);
                left++;
                if (need.containsKey(d)) {
                    if (window.get(d).equals(need.get(d))) {
                        valid--;
                    }
                    window.put(d, window.get(d) - 1);

                }

            }
        }
        return res;
    }
}

class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        //????????????26???int??????target??????????????????p??????????????????a~z??????????????????
        //???p="abc",???target???[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        //???p="bbdfeee",???target???[0,2,0,1,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        int[] target = new int[26];
        for (int i = 0; i < p.length(); i++) {
            target[p.charAt(i) - 'a']++;
        }
        //????????????????????????????????????
        //1.?????????right?????????????????????window?????????????????????????????????
        //2.???????????????????????????????????????????????????????????????????????????p???????????????????????????
        //3.??????????????????????????????????????????????????????????????????
        //4.??????????????????2???3?????????end?????????????????????s?????????
        int left = 0, right = 0;
        int[] window = new int[26];//???????????????target??????????????????????????????s???left?????????right???????????????????????????????????????????????????
        List<Integer> ans = new ArrayList<Integer>();
        while (right < s.length()) {
            window[s.charAt(right) - 'a']++;//???????????????right??????????????????s???right??????????????????????????????1
            if (right - left + 1 == p.length()) {
                if (Arrays.equals(window, target)) ans.add(left);//??????Arrays.equals????????????window?????????target???????????????????????????
                window[s.charAt(left) - 'a']--;//???????????????????????????s???left??????????????????????????????1??????1?????????????????????????????????????????????
                left++;//?????????????????????
            }
            right++;//?????????????????????
        }
        return ans;
    }

    // ???????????? + ??????;??????????????????????????????????????????p??????
    public List<Integer> findAnagrams1(String s, String p) {
        List<Integer> list = new ArrayList<>();
        int length = s.length();
        int length1 = p.length();
        if (length < length1) return list;
        // ???????????????p??????????????????,???????????????????????????
        int[] countS = new int[26];
        int[] countP = new int[26];
        for (int i = 0; i < length1; i++) {
            countP[p.charAt(i) - 'a']++;
            countS[s.charAt(i) - 'a']++;
        }
        if (Arrays.equals(countP, countS)) {
            list.add(0);
        }
        for (int i = length1; i < length; i++) {
            countS[s.charAt(i) - 'a']++;
            countS[s.charAt(i - length1) - 'a']--;// ?????????????????????????????????length1
            if (Arrays.equals(countP, countS)) {
                list.add(i - length1 + 1);//?????????????????????????????????????????????????????????+1
            }
        }
        return list;

    }

    // ??????????????????????????????????????????????????????????????????????????????
    // ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    // ?????????????????????????????????????????????????????????,???????????????????????????????????????????????????
    // ????????????????????????????????????????????????????????????????????????????????????????????????
    // ???????????????????????????????????????????????????????????????????????????
    public List<Integer> findAnagrams02(String s, String p) {
        List<Integer> list = new ArrayList<>();
        int length = p.length();
        int length1 = s.length();
        if (length > length1) return list;
        int[] countP = new int[26];
        int[] countS = new int[26];
        for (int i = 0; i < length; i++) {
            countP[p.charAt(i) - 'a']++;
        }
        int left = 0, right = 0;
        for (int i = 0; i < length1; i++) {

            if (++countS[s.charAt(i) - 'a'] > countP[s.charAt(i) - 'a']) {
                countS[s.charAt(left) - 'a']--;
                left++;
            }
            if (right - left + 1 == length) list.add(left);
        }
        return list;

    }
}


class Solution {
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int left = 0, right = 0;
        int res = 0;
        while (right < s.length()) {
            char c = s.charAt(right);
            right++;
            map.put(c, map.getOrDefault(c, 0) + 1);
            while (map.get(c) > 1) {
                char d = s.charAt(left);
                left++;
                map.put(d, map.getOrDefault(d, 0) - 1);

            }
            res = Math.max(res, right - left);
        }
        return res;
    }
}

class Solution {
    public int lengthOfLongestSubstring(String s) {
        // ????????????????????????????????????
        int[] last = new int[128];
        for (int i = 0; i < 128; i++) {
            last[i] = -1;
        }
        int n = s.length();

        int res = 0;
        int start = 0; // ??????????????????
        for (int i = 0; i < n; i++) {
            int index = s.charAt(i);
            start = Math.max(start, last[index] + 1);
            res = Math.max(res, i - start + 1);
            last[index] = i;
        }

        return res;
    }
}

class Solution {
    public boolean isFlipedString(String s1, String s2) {


        if (s1.length() != s2.length()) {
            return false;
        }
        s1 = s1 + s1;
        return s1.contains(s2);

    }
}

class Solution {
    public int maxProfit(int[] prices) {
        int maxP = 0;
        for (int i = 0; i < prices.length - 1; i++) {
            for (int j = i + 1; j < prices.length; j++) {
                if(prices[j]<=prices[i]){
                    continue;
                }
                maxP = Math.max(maxP, prices[j] - prices[i]);
            }
        }
        return maxP;
    }
}




