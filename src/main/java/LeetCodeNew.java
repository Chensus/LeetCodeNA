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
//         解题思路类似于124题, 对于任意一个节点, 如果最长同值路径包含该节点, 那么只可能是两种情况:
//         1. 其左右子树中加上该节点后所构成的同值路径中较长的那个继续向父节点回溯构成最长同值路径
//         2. 左右子树加上该节点都在最长同值路径中, 构成了最终的最长同值路径
//         需要注意因为要求同值, 所以在判断左右子树能构成的同值路径时要加入当前节点的值作为判断依据
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
//        maxl = math.max(maxl, left+right); // 路径长度为节点数减1所以此处不加1
//        if(r.val == val) // 和父节点值相同才返回以当前节点所能构成的最长通知路径长度, 否则返回0
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
//        //下标段[0, k]中，偶数下标填充[1,2,3..]
//        for(int i = 0; i <= k; i+=2){
//            res[i] = odd++;
//        }
//        //下标段[0, k]中，奇数下标填充[k + 1, k, k - 1...]
//        for(int i = 1; i <= k; i+=2){
//            res[i] = even--;
//        }
//        //下标段[k + 1, n - 1]都是顺序填充
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
//        //元素在数组内
//        for (int i = 0; i < n; i++) {
//            map.put(nums[i], math.max(map.getordefault(nums[i], n - i), n - i));
//            if (nums[i] == map.get(nums[i])) {
//                return nums[i];
//            }
//        }
//        //元素在数组外
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
//        // 防止溢出
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
//得到值重建链表
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
        // 数组大小为 amount + 1，初始值也为 amount + 1
        Arrays.fill(dp, amount + 1);

        // base case
        dp[0] = 0;
        // 外层 for 循环在遍历所有状态的所有取值
        for (int i = 0; i < dp.length; i++) {
            // 内层 for 循环在求所有选择的最小值
            for (int coin : coins) {
                // 子问题无解，跳过
                if (i - coin < 0) {
                    continue;
                }
                dp[i] = Math.min(dp[i], 1 + dp[i - coin]);
            }
        }
        return (dp[amount] == amount + 1) ? -1 : dp[amount];
    }
}
